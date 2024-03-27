import pandas as pd, numpy as np
from osgeo import gdal
from pathlib import Path
import geopandas as gpd
import urllib.request

from sbds import resamp
import sbds

image = pth/'image.tif'
resample = True # Resample data to match zoom level 19 resolution
ckpt = 'SBDS.pt'


if __name__=='__main__':
    if resample:
         # Resample data to match zoom level 19 resolution. The model was trained on this resolution, so will produce optimal results after first resampling.
        resamp(image, compress='zstd')
        image = pth/f'{image.stem}_resamp.tif'

    assert image.exists() 
    sbds.getBuildings(image, 
        box_threshold=0.0,
        )

    # Extracts building footprints from satellite imagery using a custom fine-tuned model
    # for automatic segmentation. It supports downloading new imagery for a specified area of
    # interest (AOI) or processing an existing satellite image. The extracted building footprints
    # are saved as vector data in the GeoPackage format.

    # Parameters:
    # - image: Specifies the path to a satellite image or 'download' to download imagery
    #   for the AOI defined in `extents`.
    # - outVector: Path where the extracted building footprints vector file will be saved.
    #   If 'default', a path is generated based on the input image's location and name.
    # - box_threshold: Threshold for the bounding box detection in the segmentation model.
    # - extents: Path to a file defining the AOI for downloading new imagery. Required if
    #   `image` is 'download'.
    # - source: Defines the tile source of imagery for downloading when `image` is 'download'.   
    # It can be one of the following: "OPENSTREETMAP", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID",
    #  or an HTTP URL in the format 'http://myurl.com?someargswith{x}{y}and{z}'.
    # - overwrite: If True, allows overwriting existing files when downloading or saving output files.
    # - YOLOcheckpoint: Specifies the path to the checkpoint file of the YOLO model used for
    #   building detection.

    # The function processes the satellite image, applying a segmentation model to detect buildings
    # and saving the results as vector data. It supports handling large images in patches and 
    # refines the output to include only buildings within the AOI if an `extents` file is provided.