# big = rxr.open_rasterio(tif)
# big
def resamp(tif,outtif='infer',res=0.295):
    tif19 = pth/f'{tif.stem}_resamp.tif' if outtif == 'infer' else outtif
    print(f'Resampling {tif} to resolution of {res}\n-> {tif19}')
    big = rxr.open_rasterio(tif)
    # big = shp.rast.asRioxds(tif)
    # xscale = np.abs(float(big.x[1]-big.x[0]))
    # yscale = np.abs(float(big.y[1]-big.y[0]))
    # xfreq = int(np.round(res/xscale))
    # yfreq = int(np.round(res/yscale))
    # xfreq
    unit = big.rio.crs.linear_units
    assert unit in {'meter', 'metre'}
    # Downscale to cell  size of res
    small = big.rio.reproject(big.rio.crs, resolution=(res, res))
    del big
    # small = big.resample({'x':xfreq}).mean()
    
    assert np.isclose(
        np.abs(small.x[1]-small.x[0]) ,
        np.abs(small.y[1]-small.y[0]) ,
        res
    )
    small.rio.to_raster(tif19)
    print(f'Bounced to {tif19}')
    return tif19



# big.x[1]-big.x[0], big.y[1]-big.y[0]
# 0.1016002 * 3
# xscale = np.abs(float(big.x[1]-big.x[0]))
# yscale = np.abs(float(big.y[1]-big.y[0]))
# # xfreq = int(np.round(res/xscale))
# # yfreq = int(np.round(res/yscale))
# # xfreq
# unit = big.rio.crs.linear_units
# assert unit in {'meter', 'metre'}
# res = 0.295
# # Downscale to cell  size of res
# small = big.rio.reproject(big.rio.crs, resolution=(res, res))
# # small = big.resample({'x':xfreq}).mean()
# small
# tif19 = pth/f'{tif.stem}_resamp.tif'
# small.rio.to_raster(tif19)
# small.x[1]-small.x[0], small.y[1]-small.y[0]


def get_patch_transform(original_transform, x_offset, y_offset):
    """
    Adjust the original transform for a specific patch,
    based on the offsets in the x and y directions.
    """
    new_transform = rasterio.Affine(original_transform.a, original_transform.b, original_transform.c + (original_transform.a * x_offset),
                                    original_transform.d, original_transform.e, original_transform.f + (original_transform.e * y_offset))
    return new_transform

from shapely.ops import unary_union

def merge_vector_files(file_paths, output_path):
    """
    Merges touching polygons from multiple vector files into a single output file.
    
    Parameters:
    - file_paths: Iterable of paths to the vector files.
    - output_path: Path for the output merged vector file.
    """
    # Read each vector file into a GeoDataFrame and store in a list
    gdfs = [gpd.read_file(path) for path in file_paths]
    
    # Concatenate all GeoDataFrames into a single one
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    final_gdf = combined_gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
    
    # Save the resulting GeoDataFrame to the specified output file
    final_gdf.to_file(output_path)