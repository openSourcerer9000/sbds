from pathlib import Path
import pandas as pd, numpy as np
from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr
import rasterio
import holoviews as hv
import hvplot.pandas
hv.extension("plotly")

import samgeo
import os
import torch
import leafmap
from samgeo import tms_to_geotiff
from bldg_sam import BldgSAM
g = 'geometry'
from shapely.geometry import box

pth = Path.cwd()
YOLOckpt = pth/'best.pt'

# def getBuildings(extents=None,
#     image='download',
#     outVector='default',
#     box_threshold=0.0, 
#     # showplot=True,
#     source='Satellite',
#     overwrite=True,
    # YOLOcheckpoint=YOLOckpt):
#     extents = Path(extents)
#     # Take bounding box of area of interest in WGS 84 coordinates
#     AOI = gpd.read_file(extents)
#     bbox = list(AOI.to_crs(epsg=4326).total_bounds)
#     bbox
#     # image = 'satellite19.tif'
#     if image=='download':
#         image = extents.parent/f"{extents.stem}.tif"
#         # if not image.exists():
#         a = box(*bbox).area
#         MBperDeg = 2.5/a
#         MBperDeg = 286464.2
#         mb = int(MBperDeg * a)+1

#         print(f'Downloading imagery to\n {image}\nThis will require an estimated {mb} MB...')
#         if mb>2000:
#             ans = input('OK to proceed?')
#             if ans.lower()[0] == 'n':
#                 return
#         tms_to_geotiff(output=str(image), bbox=bbox, zoom=19, source=source,overwrite=overwrite)
#     # 🤖 Automatic Segmentation with Custom Fine-Tuned Model 

#     # ckpt = Path(r'C:\Users\seanm\Docs\SegmentAnything\finetune\runs\detect\train5\weights\best.pt')
#     sam = BldgSAM(YOLOcheckpoint=YOLOcheckpoint)
#     # If you have an NVIDIA GPU and have set up CUDA correctly, this will return 'cuda'
#     # If not, this link should provide you the correct install command for your platform
#     #  https://pytorch.org/get-started/locally/
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f'Running on {device}')

#     delphic(sam,image,outVector='default',box_threshold=0.0)

    # bldgs = gpd.read_file(vector)
    # # Here we extract only the segmented buildings in our Area of Interest
    # # This technique of looking at a larger image to give us better context helps avoid 
    # bldgs = bldgs.to_crs(AOI.crs)
    # AOIpoly = AOI.dissolve().loc[0,g]
    # AOIpoly
    # # Subset to only the buildings which intersect our area of interest
    # # If needed, please review geopandas, pandas, and shapely documentation on how this works
    # bldgs = bldgs[ bldgs[g].intersects(AOIpoly) ]
    # bldgs.to_file(vector)
    # print(f'Building footprints exported to {vector}')
    # return sam.show_anns(
    #     cmap='Blues',
    #     # box_color='red',
    #     # title='Text Prompted Segmentation',
    #     blend=True,
    # )


def delphic(sam,image,outVector='default',outTIF='default',box_threshold=0.0):
    '''
    Run inference on the image and return the vectorized buildings\n
    image (Image): Input image must be a path to an image file, a numpy array, or a PIL Image.\n
    box_threshold (float): Box threshold for the prediction.
    '''
    sam.predict(image, 
        box_threshold=0)
    # sam.show_anns(
    #     cmap='Blues',
    #     box_color='red',
    #     title='Text Prompted Segmentation',
    #     blend=True,
    # )
    # <b>From the docs:</b>

    # Part of the model prediction includes setting appropriate thresholds for object detection and text association with the detected objects. These threshold values range from 0 to 1 and are set while calling the predict method of the LangSAM class.
    # <br><br>
    # `box_threshold`: This value is used for object detection in the image. A higher value makes the model more selective, identifying only the most confident object instances, leading to fewer overall detections. A lower value, conversely, makes the model more tolerant, leading to increased detections, including potentially less confident ones.
    # <br><br>
    # `text_threshold`: This value is used to associate the detected objects with the provided text prompt. A higher value requires a stronger association between the object and the text prompt, leading to more precise but potentially fewer associations. A lower value allows for looser associations, which could increase the number of associations but also introduce less precise matches.
    if outTIF=='default':
        outTIF = image.parent/f"{image.stem}-bldgs.tif"
    # Load the TIF image file and run inference
    sam.show_anns(
        cmap='Greys_r',
        add_boxes=False,
        alpha=1,
        title=f'Saved to {outTIF}',
        blend=False,
        output=str(outTIF),
    )
    vector = extents.parent/f"{image.stem}-bldgs.gpkg" if outVector=='default' else outVector
    sam.raster_to_vector(str(outTIF), str(vector))
    return vector