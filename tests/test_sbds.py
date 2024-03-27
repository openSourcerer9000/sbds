#!/usr/bin/env python

import pytest

import os
import sbds

import rioxarray as rxr
from pathlib import Path, PurePath
import pandas as pd, numpy as np
from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr
import rasterio

import samgeo
import os
import torch
import leafmap
from samgeo import tms_to_geotiff

g = 'geometry'
from shapely.geometry import box

from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from patchify import patchify
import numpy as np
import os
import sys
from PIL import Image
import shutil

extents = 'Extents.geojson'
Path(extents).write_text('''{
"type": "FeatureCollection",
"name": "extents",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": [
{ "type": "Feature", "properties": { "id": null }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 150.897542030235172, -33.940430418845317 ], [ 150.89665727452882, -33.940331218963088 ], [ 150.895367354288709, -33.940175224525859 ], [ 150.895141405202423, -33.940150582925476 ], [ 150.894953644694112, -33.940120660972539 ], [ 150.894856051661577, -33.940072257791151 ], [ 150.894772248948868, -33.939975451345795 ], [ 150.894646014482845, -33.939752796103754 ], [ 150.89443173412883, -33.93933784614434 ], [ 150.894377103246399, -33.939182074375537 ], [ 150.894362782529697, -33.939053584453426 ], [ 150.894401501504461, -33.938817725735589 ], [ 150.89466723035946, -33.938210914576167 ], [ 150.895312193009374, -33.938379009181034 ], [ 150.895718477047097, -33.938440614450499 ], [ 150.896003830587972, -33.938474937367019 ], [ 150.896189469508556, -33.938517180937566 ], [ 150.896366622078546, -33.938619269479894 ], [ 150.896457850348128, -33.938709036890103 ], [ 150.896483309400082, -33.938743359698414 ], [ 150.89668883820508, -33.93912222977842 ], [ 150.897432454681336, -33.93884786816411 ], [ 150.897587330580819, -33.938812665326836 ], [ 150.897724172985164, -33.938795063902731 ], [ 150.897908751112027, -33.938801224401573 ], [ 150.898143186548879, -33.938824106250543 ], [ 150.898190568004566, -33.938839393691907 ], [ 150.897895932517713, -33.939559068528467 ], [ 150.897542030235172, -33.940430418845317 ] ] ] } }
]
}
''')



def test_getBuildings():

    vec = 'bldgs.gpkg'
    Path(vec).unlink(missing_ok=True)
    assert not os.path.exists(vec)

    sbds.getBuildings(extents=extents,overwriteVec=True)

    assert os.path.exists('bldgs_unfiltered.gpkg')
    assert os.path.exists('bldgs.gpkg')


