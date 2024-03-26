import pandas as pd, numpy as np
from osgeo import gdal
from pathlib import Path
import geopandas as gpd
import urllib.request

from sbds import resamp
import sbds

image = pth/'475488E3341214N.tif'
resample = True # Resample data to match zoom level 19 resolution
ckpt = 'SBDS.pt'


if __name__=='__main__':
    # Download building detector model checkpoint
    ckpt = Path(ckpt)
    if not ckpt.exists():
        model_url = 'https://huggingface.co/openSourcerer9000/sbds-model/resolve/main/SBDS.pt'
        urllib.request.urlretrieve(model_url, ckpt)
    assert ckpt.exists(), ckpt

    if resample:
        resamp(image, compress='zstd')
        image = pth/f'{image.stem}_resamp.tif'

    assert image.exists() 
    sbds.getBuildings(image, 
        box_threshold=0.0,
        YOLOcheckpoint=ckpt
        overwrite=False,
        )