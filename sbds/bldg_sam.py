from osgeo import gdal
from samgeo.common import *
import os
import sys
import warnings
import argparse
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd, numpy as np
from pathlib import PurePath
from patchify import patchify, unpatchify

try:
    import rasterio
except ImportError:
    print("Installing rasterio...")
    install_package("rasterio")
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics YOLO...")
    install_package("ultralytics")
    from ultralytics import YOLO


warnings.filterwarnings("ignore")

pth = Path.cwd()
YOLOckpt = pth/'best.pt'

# Mode checkpoints
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Cache path
CACHE_PATH = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
)

import torch
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# def merge_touching_bboxes(all_boxes):
#     """
#     Merge bounding boxes that touch but do not overlap.

#     Args:
#         all_boxes (torch.Tensor): Tensor of bounding boxes in xyxy format.

#     Returns:
#         torch.Tensor: Merged bounding boxes in xyxy format.
#     """
#     # Convert all_boxes tensor to a list of Shapely box geometries
#     boxes = [box(*bbox) for bbox in all_boxes.tolist()]

#     # Create a GeoSeries from the list of boxes
#     gseries = gpd.GeoSeries(boxes)

#     # Iterate over boxes to find and merge touching but non-overlapping boxes
#     merged_boxes = []
#     while not gseries.empty:
#         # Take the first box and find boxes that touch it but do not overlap
#         current_box = gseries.iloc[0]
#         touching = gseries.touches(current_box)
#         not_overlapping = ~gseries.overlaps(current_box)
#         candidates = gseries[touching & not_overlapping]

#         # Merge the touching boxes
#         if not candidates.empty:
#             merged = unary_union([current_box, *candidates])
#             merged_boxes.append(merged)
#             gseries = gseries.drop(candidates.index)
#         else:
#             merged_boxes.append(current_box)

#         # Drop the processed box
#         gseries = gseries.drop(gseries.index[0])

#     # Convert the merged geometries back to bounding boxes
#     new_boxes = []
#     for geom in merged_boxes:
#         minx, miny, maxx, maxy = geom.bounds
#         new_boxes.append([minx, miny, maxx, maxy])

#     # Convert the list of new boxes back to a torch tensor
#     new_boxes_tensor = torch.tensor(new_boxes, dtype=all_boxes.dtype)

#     return new_boxes_tensor

import torch
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

# def significant_touch(box1, box2, min_touch_length=20):
#     """
#     Check if two boxes touch significantly along their sides.

#     Args:
#         box1, box2: Two bounding boxes in (xmin, ymin, xmax, ymax) format.
#         min_touch_length (int): Minimum length of touching sides to consider significant.

#     Returns:
#         bool: True if boxes touch significantly, False otherwise.
#     """
#     # Convert boxes to Shapely geometry
#     geom1, geom2 = box(*box1), box(*box2)

#     # Calculate intersection
#     intersection = geom1.intersection(geom2)

#     # Check if the intersection is a LineString and its length is significant
#     return intersection.geom_type == 'LineString' and intersection.length >= min_touch_length


def patch_to_image(
    array, output, crs=None, transform=None, dtype=None, compress="deflate", **kwargs
):
    """Save a NumPy array as a GeoTIFF using the projection information from an existing GeoTIFF file.

    Args:
        array (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        output (str): The path to the output image.
        source (str, optional): The path to an existing GeoTIFF file with map projection information. Defaults to None.
        dtype (np.dtype, optional): The data type of the output array. Defaults to None.
        compress (str, optional): The compression method. Can be one of the following: "deflate", "lzw", "packbits", "jpeg". Defaults to "deflate".
    """

    from PIL import Image

    if isinstance(array, str) and os.path.exists(array):
        array = cv2.imread(array)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if transform is not None:

        # Determine the minimum and maximum values in the array

        min_value = np.min(array)
        max_value = np.max(array)

        if dtype is None:
            # Determine the best dtype for the array
            if min_value >= 0 and max_value <= 1:
                dtype = np.float32
            elif min_value >= 0 and max_value <= 255:
                dtype = np.uint8
            elif min_value >= -128 and max_value <= 127:
                dtype = np.int8
            elif min_value >= 0 and max_value <= 65535:
                dtype = np.uint16
            elif min_value >= -32768 and max_value <= 32767:
                dtype = np.int16
            else:
                dtype = np.float64

        # Convert the array to the best dtype
        array = array.astype(dtype)

        # Define the GeoTIFF metadata
        if array.ndim == 2:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": 1,
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }
        elif array.ndim == 3:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": array.shape[2],
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }

        if compress is not None:
            metadata["compress"] = compress
        else:
            raise ValueError("Array must be 2D or 3D.")

        # Create a new GeoTIFF file and write the array to it
        with rasterio.open(output, "w", **metadata) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            elif array.ndim == 3:
                for i in range(array.shape[2]):
                    dst.write(array[:, :, i], i + 1)

    else:
        img = Image.fromarray(array)
        img.save(output, **kwargs)

def significant_touch(box1, box2, touch_percentage=0.5):
    """
    Check if two boxes touch significantly along their sides, based on a percentage
    of the shortest side of the involved bounding boxes.

    Args:
        box1, box2: Two bounding boxes in (xmin, ymin, xmax, ymax) format.
        touch_percentage (float): The percentage of the shortest side that must be touching.

    Returns:
        bool: True if boxes touch significantly, False otherwise.
    """
    # Convert boxes to Shapely geometry
    geom1, geom2 = box(*box1), box(*box2)

    # Calculate intersection
    intersection = geom1.intersection(geom2)

    # Determine the minimum touch length based on the shortest side of the boxes
    min_side_length = min(box1[2]-box1[0], box1[3]-box1[1], box2[2]-box2[0], box2[3]-box2[1])
    min_touch_length = touch_percentage * min_side_length

    # Check if the intersection is a LineString and its length meets the minimum requirement
    return intersection.geom_type == 'LineString' and intersection.length >= min_touch_length


def merge_touching_bboxes(all_boxes, max_merge=4, min_touch_length=20):
    """
    Merge bounding boxes that significantly touch, with constraints.

    Args:
        all_boxes (torch.Tensor): All bounding boxes in (xmin, ymin, xmax, ymax) format.
        max_merge (int): Maximum number of boxes to merge.
        min_touch_length (int): Minimum touching length to consider for merging.

    Returns:
        torch.Tensor: Tensor of merged bounding boxes.
    """
    # Initial setup
    boxes = [box(*bbox) for bbox in all_boxes.tolist()]
    merged_indices = set()
    merged_boxes = []

    # Attempt to merge boxes
    for i, bbox in enumerate(boxes):
        if i in merged_indices:
            continue  # Skip already merged boxes

        merge_candidates = [bbox]
        for j, other_bbox in enumerate(boxes):
            if j != i and j not in merged_indices:
                if significant_touch(bbox.bounds, other_bbox.bounds, min_touch_length):
                    merge_candidates.append(other_bbox)
                    merged_indices.add(j)
                    if len(merge_candidates) == max_merge:
                        break

        # Merge candidates
        if len(merge_candidates) > 1:
            unified = unary_union(merge_candidates)
            minx, miny, maxx, maxy = unified.bounds
            merged_boxes.append((minx, miny, maxx, maxy))
            merged_indices.add(i)
        elif i not in merged_indices:
            merged_boxes.append(bbox.bounds)

    # Convert merged boxes back to tensor
    merged_boxes_tensor = torch.tensor(merged_boxes, dtype=all_boxes.dtype)
    return merged_boxes_tensor



# Usage example
# merged_boxes = merge_touching_bboxes(all_boxes)





def load_model_hf(
    repo_id: str, filename: str, ckpt_config_filename: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a model from HuggingFace Model Hub.

    Args:
        repo_id (str): Repository ID on HuggingFace Model Hub.
        filename (str): Name of the model file in the repository.
        ckpt_config_filename (str): Name of the config file for the model in the repository.
        device (str): Device to load the model onto. Default is 'cpu'.

    Returns:
        torch.nn.Module: The loaded model.
    """

    cache_config_file = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_config_filename,
        force_filename=ckpt_config_filename,
    )
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)
    cache_file = hf_hub_download(
        repo_id=repo_id, filename=filename, force_filename=filename
    )
    checkpoint = torch.load(cache_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class BldgSAM:
    """
    A Segment-Anything Model (YoloSAM) class which combines a YOLOv8 fine-tuned to detect buildings and SAM.
    """

    def __init__(self, model_type="vit_h", checkpoint=None,YOLOcheckpoint=YOLOckpt):
        """Initialize the LangSAM instance.

        Args:
            model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_b.
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using building detector checkpoint {YOLOcheckpoint}')
        self.yolo = YOLO(YOLOcheckpoint)
        self.build_sam(model_type, checkpoint)

        self.source = None
        self.image = None
        self.masks = None
        self.boxes = None
        self.logits = None
        self.prediction = None

    def build_sam(self, model_type, checkpoint_url=None):
        """Build the SAM model.

        Args:
            model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_b.
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
            checkpoint_url:
        """
        if checkpoint_url is not None:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_url)
        else:
            checkpoint_url = SAM_MODELS[model_type]
            sam = sam_model_registry[model_type]()
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
            sam.load_state_dict(state_dict, strict=True)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    # def predict_yolo(self,image,box_threshold=0.4):
    #     """
    #     Run the YOLO model prediction.

    #     Args:
    #         image (Image): Input PIL Image.
    #         box_threshold (float): Box threshold for the prediction.

    #     Returns:
    #         tuple: Tuple containing boxes, logits.
    #     """
    #     thresh = box_threshold
    #     res = self.yolo.predict(image)
    #     boxes = res[0].boxes.xyxy
    #     conf = res[0].boxes.conf
    #     del res

    #     conf = conf[conf > thresh]
    #     boxes = boxes[conf > thresh]
    #     boxes = boxes.to('cpu')
    #     conf = conf.to('cpu')
    #     return boxes, conf

    def predict_yolo(self, image, box_threshold=0.24,edge_threshold=1):
        """
        Run the YOLO model prediction after splitting the image into patches, 
        accommodating images not divisible by 256x256.

        Args:
            image (Image): Input PIL Image.
            box_threshold (float): Box threshold for the prediction.

        Returns:
            tuple: Tuple containing boxes, logits.
        """
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Determine the number of patches needed
        x_patches = (image_np.shape[0] + 255) // 256  # Rounds up if not divisible
        y_patches = (image_np.shape[1] + 255) // 256  # Rounds up if not divisible

        all_boxes = []
        all_confs = []

        for i in range(x_patches):
            for j in range(y_patches):
                # Extract each patch
                x_start = i * 256
                y_start = j * 256
                x_end = min(x_start + 256, image_np.shape[0])
                y_end = min(y_start + 256, image_np.shape[1])
                patch = image_np[x_start:x_end, y_start:y_end]

                # Convert numpy array to PIL Image
                patch_pil = Image.fromarray(patch)

                # Predict with YOLO, disable prints
                sys.stdout = open(os.devnull, 'w')
                res = self.yolo.predict(patch_pil)
                sys.stdout = sys.__stdout__
                
                boxes = res[0].boxes.xyxy
                conf = res[0].boxes.conf

                conf = conf[conf > box_threshold]
                boxes = boxes[conf > box_threshold]

                # Extend boxes close to the edges of the patch
                if edge_threshold:
                    for box in boxes:
                        if box[0] - 0 < edge_threshold:  # Left edge
                            box[0] = 0
                        if box[1] - 0 < edge_threshold:  # Top edge
                            box[1] = 0
                        if 256 - box[2] < edge_threshold:  # Right edge
                            box[2] = 256
                        if 256 - box[3] < edge_threshold:  # Bottom edge
                            box[3] = 256

                # Adjust the coordinates of the boxes
                boxes[:, 0] += j * 256
                boxes[:, 1] += i * 256
                boxes[:, 2] += j * 256
                boxes[:, 3] += i * 256

                all_boxes.append(boxes)
                all_confs.append(conf)

        # Concatenate all boxes and confidences from all patches
        all_boxes = torch.cat(all_boxes, dim=0)
        all_confs = torch.cat(all_confs, dim=0)

        # all_boxes = merge_touching_bboxes(all_boxes)

        # After obtaining all_boxes:
        print(f'Postprocessing detections...')
        all_boxes = merge_touching_bboxes(all_boxes, max_merge=4, min_touch_length=.7)
        print(f'done')
        return all_boxes.cpu(), all_confs.cpu()


    # def predict_yolo(self, image, box_threshold=0.24):
    #     """
    #     Run the YOLO model prediction after splitting the image into patches.

    #     Args:
    #         image (Image): Input PIL Image.
    #         box_threshold (float): Box threshold for the prediction.

    #     Returns:
    #         tuple: Tuple containing boxes, logits.
    #     """
    #     # Convert PIL Image to numpy array
    #     image_np = np.array(image)

    #     # Split the image into 256x256 patches
    #     patches = patchify(image_np, (256, 256, 3), step=256)

    #     all_boxes = []
    #     all_confs = []

    #     for i in range(patches.shape[0]):
    #         for j in range(patches.shape[1]):
    #             # Process each patch
    #             patch = patches[i, j, 0]

    #             # Convert numpy array to PIL Image
    #             patch_pil = Image.fromarray(patch)

    #             # Predict with YOLO
    #             res = self.yolo.predict(patch_pil)
    #             boxes = res[0].boxes.xyxy
    #             conf = res[0].boxes.conf

    #             conf = conf[conf > box_threshold]
    #             boxes = boxes[conf > box_threshold]

    #             # Adjust the coordinates of the boxes
    #             boxes[:, 0] += j * 256
    #             boxes[:, 1] += i * 256
    #             boxes[:, 2] += j * 256
    #             boxes[:, 3] += i * 256

    #             all_boxes.append(boxes)
    #             all_confs.append(conf)

    #     # Concatenate all boxes and confidences from all patches
    #     all_boxes = torch.cat(all_boxes, dim=0)
    #     all_confs = torch.cat(all_confs, dim=0)

    #     return all_boxes.cpu(), all_confs.cpu()


    def predict_sam(self, image, boxes):
        """
        Run the SAM model prediction.

        Args:
            image (Image): Input PIL Image.
            boxes (torch.Tensor): Tensor of bounding boxes.

        Returns:
            Masks tensor.
        """
        image_array = np.asarray(image)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes, image_array.shape[:2]
        )
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def set_image(self, image):
        """Set the input image.

        Args:
            image (str): The path to the image file or a HTTP URL.
        """

        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image
        else:
            self.source = None

    def predict(
        self,
        image,
        box_threshold,
        edge_threshold=1,
        output=None,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        return_results=False,
        return_coords=False,
        **kwargs,
    ):
        """
        Run both YOLOv8 and SAM model prediction.

        Parameters:
            image (Image): Input image must be a path to an image file, a numpy array, or a PIL Image.
            box_threshold (float): Box threshold for the prediction.
            output (str, optional): Output path for the prediction. Defaults to None.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            return_results (bool, optional): Whether to return the results. Defaults to False.

        Returns:
            tuple: Tuple containing masks, boxes, and logits.
        """
        if isinstance(image,PurePath):
            image = str(image)
        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image

            # Load the georeferenced image
            with rasterio.open(image) as src:
                image_np = src.read().transpose(
                    (1, 2, 0)
                )  # Convert rasterio image to numpy array
                self.transform = src.transform  # Save georeferencing information
                self.crs = src.crs  # Save the Coordinate Reference System
                image_pil = Image.fromarray(
                    image_np[:, :, :3]
                )  # Convert numpy array to PIL image, excluding the alpha channel
        elif isinstance(image, np.ndarray):
            image_np = image
            image_pil = Image.fromarray(image_np)
        elif isinstance(image, Image.Image):
            image_pil = image
            image_np = np.array(image_pil)
        else:
            raise ValueError("image must be a path to an image file, a numpy array, or a PIL Image.")

        self.image = image_pil
        # return image_pil

        boxes, logits = self.predict_yolo(
            image_pil, box_threshold, edge_threshold=edge_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)

        if boxes.nelement() == 0:  # No "object" instances found
            print("No objects found in the image.")
            return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)

        self.masks = masks
        self.boxes = boxes
        self.logits = logits
        self.prediction = mask_overlay

        if return_results:
            return masks, boxes, logits

        if return_coords:
            boxlist = []
            for box in self.boxes:
                box = box.cpu().numpy()
                boxlist.append((box[0], box[1]))
            return boxlist

        return True

    def predict_batch(
        self,
        images,
        out_dir,
        text_prompt,
        box_threshold,
        text_threshold,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        merge=True,
        verbose=True,
        **kwargs,
    ):
        """
        Run both YOLOv8 and SAM model prediction for a batch of images.

        Parameters:
            images (list): List of input PIL Images.
            out_dir (str): Output directory for the prediction.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            merge (bool, optional): Whether to merge the predictions into a single GeoTIFF file. Defaults to True.
        """

        import glob

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if isinstance(images, str):
            images = list(glob.glob(os.path.join(images, "*.tif")))
            images.sort()

        if not isinstance(images, list):
            raise ValueError("images must be a list or a directory to GeoTIFF files.")

        for i, image in enumerate(images):
            basename = os.path.splitext(os.path.basename(image))[0]
            if verbose:
                print(
                    f"Processing image {str(i+1).zfill(len(str(len(images))))} of {len(images)}: {image}..."
                )
            output = os.path.join(out_dir, f"{basename}_mask.tif")
            self.predict(
                image,
                text_prompt,
                box_threshold,
                text_threshold,
                output=output,
                mask_multiplier=mask_multiplier,
                dtype=dtype,
                save_args=save_args,
                **kwargs,
            )

        if merge:
            output = os.path.join(out_dir, "merged.tif")
            merge_rasters(out_dir, output)
            if verbose:
                print(f"Saved the merged prediction to {output}.")

    def save_boxes(self, output=None, dst_crs="EPSG:4326", **kwargs):
        """Save the bounding boxes to a vector file.

        Args:
            output (str): The path to the output vector file.
            dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
            **kwargs: Additional arguments for boxes_to_vector().
        """

        if self.boxes is None:
            print("Please run predict() first.")
            return
        else:
            boxes = self.boxes.tolist()
            coords = rowcol_to_xy(self.source, boxes=boxes, dst_crs=dst_crs, **kwargs)
            if output is None:
                return boxes_to_vector(coords, self.crs, dst_crs, output)
            else:
                boxes_to_vector(coords, self.crs, dst_crs, output)

    def show_anns(
        self,
        figsize=(12, 10),
        axis="off",
        cmap="viridis",
        alpha=0.4,
        add_boxes=True,
        box_color="r",
        box_linewidth=1,
        title=None,
        output=None,
        blend=True,
        **kwargs,
    ):
        """Show the annotations (objects with random color) on the input image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            cmap (str, optional): The colormap for the annotations. Defaults to "viridis".
            alpha (float, optional): The alpha value for the annotations. Defaults to 0.4.
            add_boxes (bool, optional): Whether to show the bounding boxes. Defaults to True.
            box_color (str, optional): The color for the bounding boxes. Defaults to "r".
            box_linewidth (int, optional): The line width for the bounding boxes. Defaults to 1.
            title (str, optional): The title for the image. Defaults to None.
            output (str, optional): The path to the output image. Defaults to None.
            blend (bool, optional): Whether to show the input image. Defaults to True.
            kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().
        """

        import warnings
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        warnings.filterwarnings("ignore")

        anns = self.prediction

        if anns is None:
            print("Please run predict() first.")
            return
        elif len(anns) == 0:
            print("No objects found in the image.")
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.image)

        if add_boxes:
            for box in self.boxes:
                # Draw bounding box
                box = box.cpu().numpy()  # Convert the tensor to a numpy array
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=box_linewidth,
                    edgecolor=box_color,
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 100

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        plt.imshow(anns, cmap=cmap, alpha=alpha)

        if title is not None:
            plt.title(title)
        plt.axis(axis)

        if output is not None:
            if blend:
                plt.savefig(output, **kwargs)
            else:
                array_to_image(self.prediction, output, self.source)

    def raster_to_vector(self, image, output, simplify_tolerance=None, **kwargs):
        """Save the result to a vector file.

        Args:
            image (str): The path to the image file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_vector(image, output, simplify_tolerance=simplify_tolerance, **kwargs)

    def show_map(self, basemap="SATELLITE", out_dir=None, **kwargs):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. It can be one of the following: SATELLITE, ROADMAP, TERRAIN, HYBRID.
            out_dir (str, optional): The path to the output directory. Defaults to None.

        Returns:
            leafmap.Map: The map object.
        """
        return text_sam_gui(self, basemap=basemap, out_dir=out_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="LangSAM")
    parser.add_argument("--image", required=True, help="path to the image")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument(
        "--box_threshold", default=0.5, type=float, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", default=0.5, type=float, help="text threshold"
    )
    args = parser.parse_args()

    with rasterio.open(args.image) as src:
        image_np = src.read().transpose(
            (1, 2, 0)
        )  # Convert rasterio image to numpy array
        transform = src.transform  # Save georeferencing information
        crs = src.crs  # Save the Coordinate Reference System

    model = LangSAM()

    image_pil = Image.fromarray(
        image_np[:, :, :3]
    )  # Convert numpy array to PIL image, excluding the alpha channel
    image_np_copy = image_np.copy()  # Create a copy for modifications

    masks, boxes, logits = model.predict(
        image_pil, args.prompt, args.box_threshold, args.text_threshold
    )

    if boxes.nelement() == 0:  # No "object" instances found
        print("No objects found in the image.")
    else:
        # Create an empty image to store the mask overlays
        mask_overlay = np.zeros_like(
            image_np[..., 0], dtype=np.int64
        )  # Adjusted for single channel

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()  # Convert the tensor to a numpy array
            mask = masks[i].cpu().numpy()  # Convert the tensor to a numpy array

            # Add the mask to the mask_overlay image
            mask_overlay += (mask > 0) * (i + 1)  # Assign a unique value for each mask

    # Normalize mask_overlay to be in [0, 255]
    mask_overlay = ((mask_overlay > 0) * 255).astype(
        rasterio.uint8
    )  # Binary mask in [0, 255]

    with rasterio.open(
        "mask.tif",
        "w",
        driver="GTiff",
        height=mask_overlay.shape[0],
        width=mask_overlay.shape[1],
        count=1,
        dtype=mask_overlay.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask_overlay, 1)


# if __name__ == "__main__":
#     main()

