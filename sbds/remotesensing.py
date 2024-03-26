from pathlib import Path
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

pth = Path.cwd()
YOLOckpt = pth/'best.pt'

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
        # for simplify_tolerance in range(50):
        #     bldg.simplify(simplify_tolerance/mult).to_file(f'simp{simplify_tolerance}.gpkg')
        shutil.move(invec, invec.parent/f'{invec.stem}_unfiltered.gpkg')
        bldg.to_file(invec)
        return bldg

def getBuildings(
    image='download',
    outVector='default',
    box_threshold=0.0, 
    extents=None,
    # showplot=True,
    source='Satellite',
    overwrite=True,
    YOLOcheckpoint=YOLOckpt
    ):
    # image = 'satellite19.tif'
    vector = Path(outVector) if outVector!='default' else image.parent/f"{image.stem}-bldgs.gpkg"
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
    if image_np.shape[0]>sz or image_np.shape[1]>sz:
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
                    transform = tsfm)

                # # Remember to reset standard output to default if needed later in your script
                sys.stdout = sys.__stdout__

                if patchvec: # If buildings were detected
                    patchvecs += [patchvec]
        merge_vector_files(patchvecs, vector)
    else: # smaller size, no need to patch
        del image_np
        delphic(sam,image,vector,outTIF=outTIF,box_threshold=box_threshold)

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
    vector = Path(outVector) if outVector!='default' else image.parent/f"{image.stem}-bldgs.gpkg"
    if not overwriteVec and outVector.exists():
        print(f'{outVector} already exists. Set overwriteVec=True to overwrite.')
        return
    
    res = sam.predict(image, 
        box_threshold=0)
    if res is not None:
        if outTIF=='default':
            outTIF = image.parent/f"{image.stem}-bldgs.tif"
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