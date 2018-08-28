__version__ = '0.0.5'

import json
from collections import deque
from math import inf
from functools import reduce
import copy
from math import sqrt
import kdtree

from collections import Sequence


class TopoExtractor():

    def __init__(self, precision=1e-7, point_match_threshold=None):
        self._precision = precision

        if point_match_threshold is None:
            self._point_match_threshold = precision
        else:
            assert(point_match_threshold >= precision)
            self._point_match_threshold = point_match_threshold

        self.objects = {}

        # TODO: allow for more than 2 dimensions
        self._kdtree = kdtree.create(dimensions=2)

        self._junctions = set()
        self._neighbors = {}
        self._reset()


    def _reset(self):
        self._arcs = []
        self._bbox = None
        self._transform = None


    def topify(self, obj):
        # creates a proto-topology output. All objects will have real,
        # unmodified coordinates. If bbox's are present, they will be
        # removed

        def topify_geometry(geo):
            output = {'type': geo['type']}

            # coordinates must be tuples to easily use in sets
            # everything else should be lists to be mutable
            if geo['type'] == 'Point':
                # single coordinate alone
                output['coordinates'] = tuple(geo['coordinates'])
            elif geo['type'] in ['MultiPoint', 'LineString']:
                # an array of coordinates
                output['coordinates'] = list(map(tuple(geo['coordinates'])))
            elif geo['type'] in ['MultiLineString', 'Polygon']:
                # an array of arrays of coordinates
                output['coordinates'] = list(map(lambda c: list(map(tuple, c)), geo['coordinates']))
            elif geo['type'] == 'MultiPolygon':
                # an array of arrays of arrays of coordinates
                # this is mighty ugly in python
                output['coordinates'] = list(map(lambda p: list(map(lambda c: list(map(tuple, c)), p)), geo['coordinates']))
            elif geo['type'] in ['GeometryCollection']:
                # call recursively to figure out a geometry collection
                output['geometries'] = list(map(topify_geometry, geo['geometries']))

            # if this is a topojson object, it might have properties
            output.update({
                key: geo[key] for key in ['id', 'properties'] if key in geo
            })

            return output


        def topify_feature(feature):

            output = topify_geometry(feature['geometry'])

            # only include id and properties if they are present
            output.update({
                key: feature[key] for key in ['id', 'properties'] if key in feature
            })

            return output


        def topify_feature_collection(feature_col):
            output = {
                'type': 'GeometryCollection',
                'geometries': list(map(topify_feature, feature_col['features']))
            }

            return output


        if obj['type'] == 'FeatureCollection':
            output = topify_feature_collection(obj)
        elif obj['type'] == 'Feature':
            output = topify_feature(obj)
        else:
            output = topify_geometry(obj)

        return output


    def add_geo(self, geodict, name='default'):

        self._reset()

        if isinstance(geodict, str):
            with open(geodict, 'r') as fp:
                geodict = json.load(fp)

        obj = self.topify(geodict)
        obj = self.extract(obj)

        if name in self.objects:

            print('WARNING: topography already has object with name {}'.format(name))

            if self.objects[name]['type'] != 'GeometryCollection':
                # if what we have isn't already a GeometryCollection, we will need one
                self.objects[name] = {
                    'type': 'GeometryCollection',
                    'geometries': [self.objects[name]]
                }

            if obj['type'] == 'GeometryCollection':
                # if our new obj is a GeometryCollection, we can just concat the geometries
                self.objects[name]['geometries'].extend(obj['geometries'])
            else:
                # otherwise, we just add the whole new obj to geometries
                self.objects[name]['geometries'].add(obj)

        else:
            self.objects[name] = obj


    def get_bbox(self, obj=None):
        # computes correct bboxes for geometries. If called with no object, get
        # the bounding box for the instance by looking for bounding boxes of all
        # objects

        def combMinMax(mm_a, mm_b):
            # each set should be in (min_p, max_p)

            # to start mm_b will be `None`
            if not mm_b:
                return mm_a

            if len(mm_a[0]) != len(mm_b[0]) or len(mm_a[1]) != len(mm_b[1]) or len(mm_a[0]) != len(mm_a[1]):
                raise Exception('Attempting to combine points of conflicting dimensions\n {} vs {}'.format(mm_a, mm_b))

            min_c = list(map(min, zip(mm_a[0], mm_b[0])))
            max_c = list(map(max, zip(mm_a[1], mm_b[1])))

            return min_c, max_c


        def minMaxShape(shape):
            min_max = None

            for p in shape:
                min_max = combMinMax((p, p), min_max)

            return min_max


        def minMax2Bbox(min_max):
            bbox = []
            for i in range(len(min_max[0])):
                bbox.append(min_max[0][i])
                bbox.append(min_max[1][i])

            return bbox


        def bbox2MinMax(bbox):
            return bbox[0::2], bbox[1::2]


        def minMaxPolygon(polygon):
            # polygons MUST have the outter ring as first element
            # see https://tools.ietf.org/html/rfc7946#section-3.1.6
            return minMaxShape(polygon[0])


        def combBbox(a, b):
            return minMax2Bbox(combMinMax(bbox2MinMax(a), bbox2MinMax(b)))


        if obj is None:
            return reduce(combBbox, map(self.get_bbox, self.objects.values()))

        # if we already have a bbox, just return it
        if 'bbox' in obj:
            return obj['bbox']

        if obj['type'] == 'GeometryCollection':
            return reduce(combBbox, map(self.get_bbox, obj['geometries']))
        elif obj['type'] == 'LineString':
            return minMax2Bbox(minMaxShape(obj['coordinates']))
        elif obj['type'] == 'MultiLineString':
            return minMax2Bbox(reduce(combMinMax, map(minMaxShape, obj['coordinates'])))
        elif obj['type'] == 'Polygon':
            return minMax2Bbox(minMaxPolygon(obj['coordinates']))
        elif obj['type'] == 'MultiPolygon':
            return minMax2Bbox(reduce(combMinMax, map(minMaxPolygon, obj['coordinates'])))
        elif obj['type'] == 'MultiPoint':
            return minMax2Bbox(minMaxShape(obj['coordinates']))


    def _test_neighborhood(self, i, n):
        # test to see if a point is a junction. if yes, add it to _junctions
        # note uses point index here incase a new point has been mapped to
        # an existing point

        if i in self._junctions:
            return

        # we only need to set one neighbor group per point because a point is
        # not a junction only if all neighbors are the same
        t = self._neighbors.setdefault(i, n)
        if n != t:
            self._junctions.add(i)


    def extract(self, obj):
        # extract all points and junctions, return lists of point indexes

        obj = copy.deepcopy(obj)

        def map_point(p):
            # first map the point to our given precision, then try to identify
            # existing points within our _point_match_bound

            p = tuple(map(lambda d: round(d / self._precision) * self._precision, p))

            result = self._kdtree.search_nn(p)

            # empty tree will return None on search
            if result is not None:

                # this library kindly gives us sum squared err as well
                nearest, sse = result
                dist = sqrt(sse)

                if dist < self._point_match_threshold:
                    #print('found match ', nearest)
                    return nearest.data

            #print('inserting ', p)
            self._kdtree.add(p)
            return p


        def extract_shape(shape):
            output = []

            # all shapes must be tested for interior neighbors
            for i, p in enumerate(shape[1:-1]):

                p = map_point(p)

                # note that `i == shape.index(p) - 1` because of array slice
                n = set([shape[i], shape[i + 2]])
                self._test_neighborhood(p, n)

                output.append(p)

            return output


        def extract_line(line):
            output = []
            # first point is also by definition a junction
            i = get_point_index(line[0])
            self._junctions.add(i)
            output.append(i)

            #then we can test all interior points as normal
            output.extend(extract_shape(line))

            # the last point is also always a junction
            i = get_point_index(line[i])
            self._junctions.add(i)
            output.append(i)

            return output


        def extract_ring(ring):
            output = []
            # test neighbors of ring point
            p = map_point(ring[0])
            self._test_neighborhood(p, set([ring[1], ring[-2]]))
            output.append(p)

            output.extend(extract_shape(ring))

            # rings end on the same point they started
            output.append(p)

            return output


        def extract_polygon(polygon):
            return [extract_ring(ring) for ring in polygon]


        ## TODO: All of this is now wrong. we return idexes and need to know where we plan to send them
        if obj['type'] == 'GeometryCollection':
            obj['geometries'] = list(map(self.extract, obj['geometries']))
        elif obj['type'] == 'LineString':
            obj['coordinates'] = extract_line(obj['coordinates'])
        elif obj['type'] == 'MultiLineString':
            obj['coordinates'] = list(map(extract_line, obj['coordinates']))
        elif obj['type'] == 'Polygon':
            obj['coordinates'] = extract_polygon(obj['coordinates'])
        elif obj['type'] == 'MultiPolygon':
            obj['coordinates'] = list(map(extract_polygon, obj['coordinates']))

        return obj

    def add_arc(self, arc):
        '''compacts arcs by trying both a forward and reverse arc and returning the topojson index
        see https://github.com/topojson/topojson/wiki/Introduction#geometry-objects for more info
        on topojson indexes'''

        # search for arc in arcs index
        try:
            return self._arcs.index(arc)
        except ValueError:
            pass

        # if it isn't forward, try reversed
        try:
            # reversed returns generator
            return -1 - self._arcs.index(list(reversed(arc)))
        except ValueError:
            pass

        # otherwise, it is a new arc, add it
        self._arcs.append(arc)
        return len(self._arcs) - 1


    def cut(self, obj):

        obj = copy.deepcopy(obj)

        def cut_shape(shape):
            arc_indexes = []

            arc_start = 0
            for i, p in enumerate(shape[1:-1]):
                if p in self._junctions:
                    # note that `i == shape.index(p) - 1` because of array slice
                    arc = shape[arc_start:i + 2]
                    arc_indexes.append(self.add_arc(arc))
                    arc_start = i + 1

            #once we get to the end of the ring, put in the final arc
            arc_indexes.append(self.add_arc(shape[arc_start:]))

            return arc_indexes


        def cut_ring(ring):

            # rotate ring to first junction, or leave alone if unconnected
            for offset, p in enumerate(ring):
                if p in self._junctions:
                    ring = ring[offset:-1] + ring[:offset+1]
                    break

            return cut_shape(ring)

        def cut_polygon(polygon):
            return list(map(cut_ring, polygon))

        if obj['type'] == 'GeometryCollection':
            obj['geometries'] = list(map(self.cut, obj['geometries']))
        elif obj['type'] == 'LineString':
            obj['arcs'] = cut_shape(obj['coordinates'])
        elif obj['type'] == 'MultiLineString':
            obj['arcs'] = list(map(cut_shape, obj['coordinates']))
        elif obj['type'] == 'Polygon':
            obj['arcs'] = cut_polygon(obj['coordinates'])
        elif obj['type'] == 'MultiPolygon':
            obj['arcs'] = list(map(cut_polygon, obj['coordinates']))

        # if we still have coordinates, pop it
        obj.pop('coordinates', None)

        return obj


    def get_transform(self, bbox=None):
        if bbox is None:
            bbox = self.get_bbox()

        translate = bbox[0::2]
        scale = [self._precision] * len(translate)

        return {
            'translate': translate,
            'scale': scale
        }


    def get_delta_arcs(self, transform=None):
        # quantized topographies must have a `transform` property at the top
        # level, and all arcs must be delta-encoded

        if transform is None:
            transform = self.get_transform()

        translate = transform['translate']
        scale = transform['scale']

        def transform_point(p):
            return list(map(lambda v, t, s: round((v - t) / s), p, translate, scale))

        #def add_delta(arc, p):
        #    return arc + [list(map(lambda x, y: x - y, transform_point(p), arc[-1]))]

        def delta_arc(arc):
            arc = list(map(transform_point, arc))
            output = copy.deepcopy(arc)

            for i in range(1,len(arc)):
                output[i] = list(map(lambda x, y: x - y, arc[i], arc[i - 1]))

            return output

        return list(map(delta_arc, self._arcs))


    def get_topo(self):
        topo = {
            'type': 'Topology',
            'objects': {},
        }

        for name, obj in self.objects.items():

            # find out the bbox before we cut up the obj
            bbox = self.get_bbox(obj)

            # .cut() modifies self to add arcs, and returns a new obj without
            # coordinates and instead arc indexes
            obj = self.cut(obj)

            topo['objects'][name] = obj
            topo['objects'][name]['bbox'] = bbox

        bbox = self.get_bbox()
        transform = self.get_transform(bbox)
        arcs = self.get_delta_arcs(transform)

        topo.update({
            'bbox': bbox,
            'transform': transform,
            'arcs': arcs
        })

        return topo

    def dumps(self, *args, **kwargs):
        topo = self.get_topo()
        return json.dumps(topo, *args, **kwargs)

    def dump(self, *args, **kwargs):
        topo = self.get_topo()
        return json.dump(topo, *args, **kwargs)
