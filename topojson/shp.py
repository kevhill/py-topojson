import os
import shapefile
import json
from pyproj import Proj, transform

from collections import Sequence

def recursive_map(func, seq):
    '''From https://stackoverflow.com/a/42095505/1701722'''

    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(func, item))
        else:
            yield func(item)

def shp2geodict(shp_reader, filter_fields=None, precision=6, proj=None, proj_to=None):

    if isinstance(shp_reader, str):
        shp_reader = shapefile.Reader(shp_reader)

    if proj is not None:
        proj_from = proj
        if proj_to is None:
            proj_to = Proj(init='EPSG:4326')

    field_names = [field[0] for field in shp_reader.fields[1:]]

    features = []
    for sr in shp_reader.shapeRecords():
        props = dict(zip(field_names, sr.record))
        if filter_fields:
            props = {field: props[field] for field in filter_fields}

        geom = sr.shape.__geo_interface__

        # fix any ugly floats in the shp coordinates
        coords = []
        for ring in geom['coordinates']:
            output_ring = []
            for point in ring:
                if proj is not None:
                    point = transform(proj_from, proj_to, point[0], point[1])

                try:
                    output_ring.append(list(recursive_map(lambda x: round(x, precision), point)))
                except:
                    print('wtf', point)

            coords.append(output_ring)

        geom['coordinates'] = coords

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
            "bbox": list(map(lambda x: round(x, precision), sr.shape.bbox)),
        })

    return {
        "type": "FeatureCollection",
        "bbox": list(map(lambda x: round(x, precision), shp_reader.bbox)),
        "features": features
    }
