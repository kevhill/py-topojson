import os
import shapefile
import json

from collections import Sequence

def recursive_map(func, seq):
    '''From https://stackoverflow.com/a/42095505/1701722'''

    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(func, item))
        else:
            yield func(item)

def shp2geodict(shp_reader, filter_fields=None, precsision=6):

    if isinstance(shp_reader, str):
        shp_reader = shapefile.Reader(shp_reader)

    field_names = [field[0] for field in shp_reader.fields[1:]]

    features = []
    for sr in shp_reader.shapeRecords():
        props = dict(zip(field_names, sr.record))
        if filter_fields:
            props = {field: props[field] for field in filter_fields}

        geom = sr.shape.__geo_interface__

        # fix any ugly floats in the shp coordinates
        geom['coordinates'] = list(recursive_map(lambda x: round(x, precsision), geom['coordinates']))

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
            "bbox": list(map(lambda x: round(x, precsision), sr.shape.bbox)),
        })

    return {
        "type": "FeatureCollection",
        "bbox": list(map(lambda x: round(x, precsision), shp_reader.bbox)),
        "features": features
    }
