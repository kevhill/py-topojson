__version__ = '0.0.2'

import json
from collections import deque
from math import inf
from functools import reduce
import copy

from collections import Sequence

class TopoExtractor():

    def __init__(self, precision=1e-7):
        self._precision = precision

        self._junctions = set()
        self._neighbors = dict()

        self.objects = {}
        self.arcs = []
        self.bbox = []


    def set_geo_bbox(self, obj):
        # recomputes correct bboxes for geometries

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

        if obj['type'] == 'GeometryCollection':
            deque(map(self.set_geo_bbox, obj['geometries']))
            obj['bbox'] = reduce(combBbox, map(lambda g: g['bbox'], obj['geometries']))
        elif obj['type'] == 'LineString':
            obj['bbox'] = minMax2Bbox(minMaxShape(obj['arcs']))
        elif obj['type'] == 'MultiLineString':
            obj['bbox'] = minMax2Bbox(reduce(combMinMax, map(minMaxShape, obj['arcs'])))
        elif obj['type'] == 'Polygon':
            obj['bbox'] = minMax2Bbox(minMaxPolygon(obj['arcs']))
        elif obj['type'] == 'MultiPolygon':
            obj['bbox'] = minMax2Bbox(reduce(combMinMax, map(minMaxPolygon, obj['arcs'])))
        elif obj['type'] == 'MultiPoint':
            obj['bbox'] = minMax2Bbox(minMaxShape(obj['coordinates']))
        elif obj['type'] == 'Topology':
            obj['bbox'] = reduce(combBbox, map(lambda o: o['bbox'], obj['objects'].values()))


    def geomify(self, obj):

        def recursive_tuple(seq):
            for item in seq:
                if isinstance(item, Sequence):
                    yield tuple(recursive_tuple(item))
                else:
                    yield item

        def get_if_present(d, keys):
            if isinstance(keys, str):
                keys = [keys]

            return {
                key: d[key] for key in keys if key in d
            }

        def geomifyGeometry(geo):
            output = {'type': geo['type']}

            if geo['type'] in ['GeometryCollection']:
                output['geometries'] = list(map(geomifyGeometry, geo['geometries']))
            elif geo['type'] in ['Point', 'MultiPoint']:
                output['coordinates'] = geo['coordinates']
            else:
                output['arcs'] = list(recursive_tuple(geo['coordinates']))

            return output

        def geomifyFeature(feature):
            output = geomifyGeometry(feature['geometry'])
            output.update(get_if_present(feature, ['id', 'properties']))

            return output

        def geomifyFeatureCollection(featureCol):
            output = {
                'type': 'GeometryCollection',
                'geometries': list(map(geomifyFeature, featureCol['features']))
            }

            return output

        if obj['type'] == 'FeatureCollection':
            output = geomifyFeatureCollection(obj)
        elif obj['type'] == 'Feature':
            output = geomifyFeature(obj)
        else:
            output = geomifyGeometry(obj)

        self.set_geo_bbox(output)
        return output


    def test_neighborhood(self, p, n):
        # test to see if a point is already a junction
        if p in self._junctions:
            return

        # we only need to set one neighbor group per point because a point is
        # not a junction only if all neighbors are the same
        t = self._neighbors.setdefault(p, n)
        if n != t:
            self._junctions.add(p)


    def extract(self, obj):
        # extract all the junctions

        def extractShape(shape):
            # all shapes must be tested for interior neighbors
            for i, p in enumerate(shape[1:-1]):
                # note that `i == shape.index(p) - 1` because of array slice
                n = set([shape[i], shape[i + 2]])
                self.test_neighborhood(p, n)


        def extractLine(line):
            # add first and last point to junctions
            self._junctions.add(line[0])
            self._junctions.add(line[-1])

            extractShape(line)


        def extractRing(ring):
            # test neighbors of ring point
            self.test_neighborhood(ring[0], set([ring[1], ring[-2]]))

            extractShape(ring)


        def extractPolygon(polygon):
            for ring in polygon:
                extractRing(ring)


        if obj['type'] == 'GeometryCollection':
            deque(map(self.extract, obj['geometries']))
        elif obj['type'] == 'LineString':
            extractLine(obj['arcs'])
        elif obj['type'] == 'MultiLineString':
            deque(map(extractLine, obj['arcs']))
        elif obj['type'] == 'Polygon':
            extractPolygon(obj['arcs'])
        elif obj['type'] == 'MultiPolygon':
            deque(map(extractPolygon, obj['arcs']))


    def compact(self, arc):
        '''compacts arcs by trying both a forward and reverse arc and returningthe topojson index
        see https://github.com/topojson/topojson/wiki/Introduction#geometry-objects for more info
        on topojson indexes'''
        try:
            return self.arcs.index(arc)
        except ValueError:
            pass

        try:
            # map all the arcs to lists, as they made be mutated later if we add
            # more geojson sources
            return -1 - self.arcs.index(list(reversed(arc)))
        except ValueError:
            pass

        self.arcs.append(list(arc))
        return len(self.arcs) - 1


    def cut(self, obj):

        def cutShape(shape):
            arc_indexes = []

            arc_start = 0
            for i, p in enumerate(shape[1:-1]):
                if p in self._junctions:
                    # note that `i == shape.index(p) - 1` because of array slice
                    arc = shape[arc_start:i + 2]
                    arc_indexes.append(self.compact(arc))
                    arc_start = i + 1

            #once we get to the end of the ring, put in the final arc
            arc_indexes.append(self.compact(shape[arc_start:]))

            return arc_indexes


        def cutRing(ring):

            # rotate ring to first junction, or leave alone if unconnected
            for offset, p in enumerate(ring):
                if p in self._junctions:
                    ring = ring[offset:-1] + ring[:offset+1]
                    break

            return cutShape(ring)

        def cutPolygon(polygon):
            return list(map(cutRing, polygon))


        if obj['type'] == 'GeometryCollection':
            deque(map(self.cut, obj['geometries']))
        elif obj['type'] == 'LineString':
            obj['arcs'] = cutShape(obj['arcs'])
        elif obj['type'] == 'MultiLineString':
            obj['arcs'] = list(map(cutShape, obj['arcs']))
        elif obj['type'] == 'Polygon':
            obj['arcs'] = cutPolygon(obj['arcs'])
        elif obj['type'] == 'MultiPolygon':
            obj['arcs'] = list(map(cutPolygon, obj['arcs']))


    def add(self, geodict, name='default'):

        if name in self.objects:
            raise ValueError('object {} already exists in topography'.format(name))

        if isinstance(geodict, str):
            with open(geodict, 'r') as fp:
                geodict = json.load(fp)

        self.objects[name] = self.geomify(geodict)
        self.extract(self.objects[name])
        self.cut(self.objects[name])

    def quantize(self, topo, n=None, scale=None):
        # quantized topographies must have a `transform` property at the top
        # level, and all arcs must be delta-encoded

        bbox = topo['bbox']

        # the translation is just the minimum of the bbox
        translate = bbox[0::2]

        if scale is None and n is None:
            # If no scale info is provided, default to 1e-6 degree, which should
            # be lossless for most applications but still reduce space signicantly
            scale = [self._precision] * len(translate)
        elif scale is None:
            # If we just have an n, use that to compute optimal scale given our
            # bbox range
            max_p = bbox[1::2]
            d = list(map(lambda min_p, max_p: max_p - min_p, translate, max_p))

            scale = list(map(lambda x: x / (n-1), d))

        def transformPoint(p):
            return list(map(lambda v, t, s: round((v - t) / s), p, translate, scale))

        def addDelta(arc, p):
            return arc + [list(map(lambda x, y: x - y, transformPoint(p), arc[-1]))]

        def quantizeArc(arc):
            arc = list(map(transformPoint, arc))
            output = copy.deepcopy(arc)

            for i in range(1,len(arc)):
                output[i] = list(map(lambda x, y: x - y, arc[i], arc[i - 1]))

            return output


        topo.update({
            'arcs': list(map(quantizeArc, topo['arcs'])),
            'transform': {
                'translate': translate,
                'scale': scale
            }
        })

        return topo

    def get_topo(self):
        topo = {
            'type': 'Topology',
            'objects': self.objects,
            'arcs': self.arcs
        }
        self.set_geo_bbox(topo)

        return self.quantize(topo)

    def dumps(self, *args, **kwargs):
        topo = self.get_topo()
        return json.dumps(topo, *args, **kwargs)

    def dump(self, *args, **kwargs):
        topo = self.get_topo()
        return json.dump(topo, *args, **kwargs)
