from pathlib import Path, PurePath
import pandas as pd, numpy as np
try:
    from osgeo import gdal
except Exception as e:
    print(e)
import geopandas as gpd
import rioxarray as rxr
import rasterio

import samgeo
import os
import torch
import leafmap
from samgeo import tms_to_geotiff
try:
    from bldg_sam import BldgSAM, patch_to_image
    from util import *
except:
    from .bldg_sam import BldgSAM, patch_to_image
    from .util import *
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

pth = Path.cwd()
YOLOckpt = pth/'SBDS.pt'

def BLDGcleanup(invec,areathreshold = 150, # don't forget the tiny houses!
# simplify_tolerance = 4
    ):
    bldg = gpd.read_file(invec)
    m2f = 3.28084
    unitz = bldg.crs.axis_info[0].unit_name
    if unitz not in {'degree','deg'}:
        a = bldg.area
        mult = m2f if unitz in {'metre', 'meter'} else 1
        a = a*(mult**2)
        bldg = bldg[a>areathreshold]
        #     bldg.simplify(simplify_tolerance/mult).to_file(f'simp{simplify_tolerance}.gpkg')
        shutil.move(invec, invec.parent/f'{invec.stem}_unfiltered.gpkg')
        bldg.to_file(invec)
        return bldg
    else:
        print('No filtering performed. Please reproject to a local coordinate system for area filtering, or filter out small structures in your own postprocessing')
        return bldg
        
def getBuildings(
    image='download',
    outVector='default',
    box_threshold=0.0, 
    extents=None,
    # showplot=True,
    source='Satellite',
    overwrite=True,
    overwriteVec=True,
    YOLOcheckpoint=YOLOckpt
    ):
    ''' # Get buildings
    Extracts building footprints from satellite imagery using a custom fine-tuned model
    for automatic segmentation. It supports downloading new imagery for a specified area of
    interest (AOI) or processing an existing satellite image. The extracted building footprints
    are saved as vector data in the GeoPackage format.

    Parameters:
    - image: Specifies the path to a satellite image or 'download' to download imagery
      for the AOI defined in `extents`.
    - outVector: Path where the extracted building footprints vector file will be saved.
      If 'default', a path is generated based on the input image's location and name.
    - box_threshold: Threshold for the bounding box detection in the segmentation model.
    - extents: Path to a file defining the AOI for downloading new imagery. Required if
      `image` is 'download'.
    - source: Defines the tile source of imagery for downloading when `image` is 'download'.   
    It can be one of the following: "OPENSTREETMAP", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID",
     or an HTTP URL in the format 'http://myurl.com?someargswith{x}{y}and{z}'.
    - overwrite: If True, allows overwriting existing files when downloading or saving output files.
    - YOLOcheckpoint: Specifies the path to the checkpoint file of the YOLO model used for
      building detection.

    The function processes the satellite image, applying a segmentation model to detect buildings
    and saving the results as vector data. It supports handling large images in patches and 
    refines the output to include only buildings within the AOI if an `extents` file is provided.
    '''
    # image = 'satellite19.tif'
    if image=='download':
        assert extents is not None, 'Please provide a path to the area of interest when downloading new imagery'
        extents = Path(extents)
        # Take bounding box of area of interest in WGS 84 coordinates
        AOI = gpd.read_file(extents)
        bbox = list(AOI.to_crs(epsg=4326).total_bounds)
        bbox
        image = extents.parent/f"{extents.stem}.tif"
        # if not image.exists():
        a = box(*bbox).area
        MBperDeg = 2.5/a
        MBperDeg = 286464.2
        mb = int(MBperDeg * a)+1

        print(f'Downloading imagery to\n {image}\nThis will require an estimated {mb} MB...')
        if mb>2000:
            ans = input('OK to proceed?')
            if ans.lower()[0] == 'n':
                return
        tms_to_geotiff(output=str(image), bbox=bbox, zoom=19, source=source,overwrite=overwrite)
    pth = Path(image).parent
    vector = Path(outVector) if outVector!='default' else pth/f"{Path(image).stem}-bldgs.gpkg"

    # 🤖 Automatic Segmentation with Custom Fine-Tuned Model 

    # ckpt = Path(r'C:\Users\seanm\Docs\SegmentAnything\finetune\runs\detect\train5\weights\best.pt')
    sam = BldgSAM(YOLOcheckpoint=YOLOcheckpoint)
    # If you have an NVIDIA GPU and have set up CUDA correctly, this will return 'cuda'
    # If not, this link should provide you the correct install command for your platform
    #  https://pytorch.org/get-started/locally/
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device}')
    # Load the georeferenced image
    with rasterio.open(image) as src:
        image_np = src.read().transpose((1, 2, 0))  # Convert rasterio image to numpy array
        ogtsfm = src.transform  # Save georeferencing information    
        ogcrs = src.crs
    print('opened')

    outTIF=pth/'temp.tif'
    sz = 1024 # Sam was trained on this resolution
    if image_np.shape[0]>sz and image_np.shape[1]>sz:
        # split it up
        patches = patchify(image_np, (sz, sz, 3), step=sz)
        patchvecs = []
        for i in tqdm(range(patches.shape[0])):
            for j in range(patches.shape[1]):
                patch = patches[i,j,0,:,:,:]
                
                tsfm = get_patch_transform(ogtsfm, j*sz, i*sz)

                # # Redirect standard output
                sys.stdout = open(os.devnull, 'w')

                patchvec = delphic(sam,patches[i,j,0,:,:,:],
                    outVector=pth/f'bldg{i}_{j}.gpkg',
                    outTIF=outTIF,
                    box_threshold=box_threshold,
                    crs = ogcrs,
                    transform = tsfm,
                    overwriteVec=overwriteVec)

                # # Remember to reset standard output to default if needed later in your script
                sys.stdout = sys.__stdout__

                if patchvec: # If buildings were detected
                    patchvecs += [patchvec]
        merge_vector_files(patchvecs, vector)
    else: # smaller size, no need to patch
        del image_np
        delphic(sam,image,vector,outTIF=outTIF,box_threshold=box_threshold,overwriteVec=overwriteVec)

    if extents:
        bldgs = gpd.read_file(vector)
        # Here we extract only the segmented buildings in our Area of Interest
        # This technique of looking at a larger image to give us better context helps avoid 
        bldgs = bldgs.to_crs(AOI.crs)
        AOIpoly = AOI.dissolve().loc[0,g]
        AOIpoly
        # Subset to only the buildings which intersect our area of interest
        # If needed, please review geopandas, pandas, and shapely documentation on how this works
        bldgs = bldgs[ bldgs[g].intersects(AOIpoly) ]
        bldgs.to_file(vector)
        print(f'Building footprints exported to {vector}')
        return sam.show_anns(
            cmap='Blues',
            # add_boxes=False,
            # box_color='red',
            # title='Text Prompted Segmentation',
            blend=True,
        )
    else:
        print(f'Building footprints exported to {vector}')

    BLDGcleanup(vector)




def delphic(sam,image,outVector='default',outTIF='default',box_threshold=0.0,
    crs=None,transform=None,
    overwriteVec=False):
    '''
    Run inference on the image and return the vectorized buildings\n
    image (Image): Input image must be a path to an image file, a numpy array, or a PIL Image.\n
    box_threshold (float): Box threshold for the prediction.
    '''
    # vector = extents.parent/f"{image.stem}-bldgs.gpkg" if outVector=='default' else outVector
    ispth = isinstance(image,PurePath)
    pth = image.parent if ispth else Path.cwd()
    vector = Path(outVector) if outVector!='default' else pth/f"{(image.stem+'-') if ispth else ''}bldgs.gpkg"
    if not overwriteVec and outVector.exists():
        print(f'{outVector} already exists. Set overwriteVec=True to overwrite.')
        return
    
    res = sam.predict(image, 
        box_threshold=box_threshold)
    if res is not None:
        if outTIF=='default':
            outTIF = pth/f"{(image.stem+'-') if ispth else ''}bldgs.tif"
        # Load the TIF image file and run inference

        if transform is None:
            sam.show_anns(
                cmap='Greys_r',
                add_boxes=False,
                alpha=1,
                title=f'Saved to {outTIF}',
                blend=False,
                output=str(outTIF),
            )
        else:
            patch_to_image(sam.prediction, outTIF,crs,transform)
        
        sam.raster_to_vector(str(outTIF), str(vector))
        return vector