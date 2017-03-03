from psycopg2.extras import RealDictCursor
from tilequeue.postgresql import DBAffinityConnectionsNoLimit
from tilequeue.tile import calc_meters_per_pixel_dim
from tilequeue.tile import coord_to_mercator_bounds
from tilequeue.transform import calculate_padded_bounds
import threading
#from lru import LRUCacheDict
from pylru import lrucache
import sys
from shapely import wkb
from shapely.geometry import box as Box


def generate_query(start_zoom, template, bounds, zoom):
    if zoom < start_zoom:
        return None
    query = template.render(bounds=bounds, zoom=zoom)
    return query


class JinjaQueryGenerator(object):

    def __init__(self, template, start_zoom):
        self.template = template
        self.start_zoom = start_zoom

    def __call__(self, bounds, zoom):
        return generate_query(self.start_zoom, self.template, bounds, zoom)


class DevJinjaQueryGenerator(object):

    def __init__(self, environment, template_name, start_zoom):
        self.environment = environment
        self.template_name = template_name
        self.start_zoom = start_zoom

    def __call__(self, bounds, zoom):
        template = self.environment.get_template(self.template_name)
        return generate_query(self.start_zoom, template, bounds, zoom)


def jinja_filter_geometry(value):
    return 'ST_AsBinary(%s)' % value


def jinja_filter_bbox_filter(bounds, geometry_col_name, srid=3857):
    min_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[0], bounds[1])
    max_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[2], bounds[3])
    bbox_no_srid = 'ST_MakeBox2D(%s, %s)' % (min_point, max_point)
    bbox = 'ST_SetSrid(%s, %d)' % (bbox_no_srid, srid)
    bbox_filter = '%s && %s' % (geometry_col_name, bbox)
    return bbox_filter


def jinja_filter_bbox_intersection(bounds, geometry_col_name, srid=3857):
    min_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[0], bounds[1])
    max_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[2], bounds[3])
    bbox_no_srid = 'ST_MakeBox2D(%s, %s)' % (min_point, max_point)
    bbox = 'ST_SetSrid(%s, %d)' % (bbox_no_srid, srid)
    bbox_intersection = 'st_intersection(%s, %s)' % (geometry_col_name, bbox)
    return bbox_intersection


def jinja_filter_bbox_padded_intersection(
        bounds, geometry_col_name, pad_factor=1.1, srid=3857):
    padded_bounds = calculate_padded_bounds(pad_factor, bounds)
    return jinja_filter_bbox_intersection(
        padded_bounds.bounds, geometry_col_name, srid)


def jinja_filter_bbox(bounds, srid=3857):
    min_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[0], bounds[1])
    max_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[2], bounds[3])
    bbox_no_srid = 'ST_MakeBox2D(%s, %s)' % (min_point, max_point)
    bbox = 'ST_SetSrid(%s, %d)' % (bbox_no_srid, srid)
    return bbox


def jinja_filter_bbox_overlaps(bounds, geometry_col_name, srid=3857):
    min_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[0], bounds[1])
    max_point = 'ST_MakePoint(%.12f, %.12f)' % (bounds[2], bounds[3])
    bbox_no_srid = 'ST_MakeBox2D(%s, %s)' % (min_point, max_point)
    bbox = 'ST_SetSrid(%s, %d)' % (bbox_no_srid, srid)
    bbox_filter = \
        '((%(col)s && %(bbox)s) AND st_overlaps(%(col)s, %(bbox)s))' \
        % dict(col=geometry_col_name, bbox=bbox)
    return bbox_filter


def base_feature_query(unpadded_bounds, layer_datum, zoom):
    meters_per_pixel_dim = calc_meters_per_pixel_dim(zoom)
    query_bounds_pad_fn = layer_datum['query_bounds_pad_fn']
    padded_bounds = query_bounds_pad_fn(
        unpadded_bounds, meters_per_pixel_dim)
    query_generator = layer_datum['query_generator']
    query = query_generator(padded_bounds, zoom)
    return (query, padded_bounds)

def execute_query(conn, query):
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        return list(cursor.fetchall())
    except:
        # If any exception occurs during query execution, close the
        # connection to ensure it is not in an invalid state. The
        # connection pool knows to create new connections to replace
        # those that are closed
        try:
            conn.close()
        except:
            pass
        raise


def trim_layer_datum(layer_datum):
    layer_datum_result = dict(
        [(k, v) for k, v in layer_datum.items()
         if k not in ('query_generator', 'query_bounds_pad_fn')])
    return layer_datum_result

def read_row(row):
    result = {}
    for k, v in row.items():
        if isinstance(v, buffer):
            v = bytes(v)
        if v is not None:
            result[k] = v
    return result

class DataFetcher(object):

    def __init__(self, conn_info, layer_data, io_pool, n_conn):
        self.conn_info = dict(conn_info)
        self.layer_data = layer_data
        self.io_pool = io_pool

        self.dbnames = self.conn_info.pop('dbnames')
        self.dbnames_query_index = 0
        self.sql_conn_pool = DBAffinityConnectionsNoLimit(
            self.dbnames, self.conn_info)
        self.n_conn = n_conn

        #TODO: make cache size configurable per-layer
        self.caches = {
            #layer_datum['name']:(threading.Lock(), LRUCacheDict(max_size=10000, expiration=5*60 ))
            layer_datum['name']:(threading.Lock(), lrucache(10000))
            for layer_datum in layer_data
        }

    def fetch_results_for_layer(self, conn, layer_datum, unpadded_bounds, coord):
        #in theory, this can return None for base query? to investigate.
        #that's what enqueue_queries thinks.
        #if so, we'd return an empty layer
        base, padded_bounds = base_feature_query(unpadded_bounds, layer_datum, coord.zoom)
        id_query = "select __id__ from (%s) x" % base
        ids = [int(r['__id__']) for r in execute_query(conn, id_query)]

        layer_lock, layer_cache = self.caches[layer_datum['name']]
        with layer_lock:
            cached_items = { k:layer_cache[k]['item'].copy() for k in ids if k in layer_cache }
            ids_to_fetch = set(ids) - set(cached_items.keys())


        if len(ids_to_fetch) > 0:
            fetch_query = "select * from (%s) x where __id__ in (%s)" % (base, ','.join(map(str,ids_to_fetch)))
            results = execute_query(conn, fetch_query)
            result_items = map(read_row, results)
            db_items = {item['__id__']:item for item in result_items}
        else:
            db_items = {}

        all_items = [(cached_items.get(i) or db_items.get(i)) for i in ids]

        #update cache and set clipped geometry
        #this is hacky, datafetcher isn't supposed to know about geometry
        #also, without a per-layer config, we don't know if we really want to return clipped geometry or not
        #it's also dangerous, putting a (potentially) CPU-heavy task blocking the io_thread
        bbox = Box(*padded_bounds['polygon'])
        parent_coords = [
            (coord.column/2**z, coord.row/2**z, coord.zoom-z) for z in xrange(0,coord.zoom+1)
        ] + ["full"]
        contained = 0
        clipped = 0
        empty = 0
        with layer_lock:
            for item in all_items:
                if not item.get('__maycache__'):
                    continue
                key = item['__id__']
                if not key in layer_cache:
                    layer_cache[key] = { 'item': item.copy(), 'clips': {'full': wkb.loads(item['__geometry__']) } }
                cached_clips = layer_cache[key]["clips"]
                closest_parent = next(cached_clips[c] for c in parent_coords if c in cached_clips)
                if bbox.contains(closest_parent) or closest_parent.is_empty:
                    contained += 1
                    item['__geometry__'] = closest_parent.wkb
                elif bbox.intersects(closest_parent):
                    clipped += 1
                    clipped_geom = closest_parent.intersection(bbox)
                    cached_clips[(coord.column, coord.row, coord.zoom)] = clipped_geom
                    item['__geometry__'] = clipped_geom.wkb

        feature_layer = dict(
            name=layer_datum['name'],
            features=all_items,
            layer_datum=trim_layer_datum(layer_datum),
            padded_bounds=padded_bounds,
        )
        print "for %r, %d contained %d clipped %d non-intersect" % (coord, contained, clipped, len(ids) - contained - clipped)

        return feature_layer

    def __call__(self, coord, layer_data=None):
        if layer_data is None:
            layer_data = self.layer_data
        unpadded_bounds = coord_to_mercator_bounds(coord)

        #note: this isn't really a pool...
        def fetch_layer(layer_datum):
            sql_conn = self.sql_conn_pool.get_conns(1)[0]
            try:
                return self.fetch_results_for_layer(
                    sql_conn,
                    layer_datum,
                    unpadded_bounds,
                    coord
                )
            finally:
                self.sql_conn_pool.put_conns([sql_conn])

        #feature_layers = self.io_pool.map( fetch_layer, layer_data )
        feature_layers = map( fetch_layer, layer_data )

        #the previous version assumes there will be at least one layer, and padded bounds are effectively the same for all, apparently
        padded_bounds = feature_layers[0]['padded_bounds']
        return dict(
            feature_layers=feature_layers,
            unpadded_bounds=unpadded_bounds,
            padded_bounds=padded_bounds,
        )
