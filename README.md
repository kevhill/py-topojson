# py-topojson
A package for creating and modifying topojson maps 

## Overview
This package is based on the [reference topojson implementation in node.js](https://github.com/topojson/topojson).
It follows a similar logic outlined [in this blogpost](https://bost.ocks.org/mike/topology/) but simplifies things
a bit by finding junctions first, then extracting and cutting in a single step. Other minor changes have been made
from the reference implementation for ease of use and to make the library more pythonic.

## Usage
Basic usage is by creating an extractor object, then using the `.add` and `.dump` methods on the extractor to add
geojson and push that data to a file object.

```python
import json
from topojson import TopoExtractor

e = TopoExtractor(1e-6)

with open('./geo.json', 'r') as fp:
    geojson = json.load(fp)

e.add(geojson)

with open('./topo.json', 'w') as fp:
    e.dump(fp)
```

The only input to the extractor constructor is a percision parameter. Instead of finding the percision and offset from the
data as the reference implementation, this library takes an explicit percision parameter to make things both more interpretable
in the raw json file, and to simplify the computation needed to ensure all arc data are integers.


## Roadmap
* Allow multiple geojson files to be added as separate objects, and reuse arcs across objects if possible. Biggest issue
here seems to be how to match arcs
* Better documentation
* Utilities for working with object properties
