from copy import deepcopy
from math import ceil
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .base_methods import read_img_as_numpy



def extract_crops(img: np.ndarray, boxes: np.ndarray, channels_last: bool = True) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes
    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    if channels_last:
        return deepcopy([img[box[1] : box[3], box[0] : box[2]] for box in _boxes])

    return deepcopy([img[:, box[1] : box[3], box[0] : box[2]] for box in _boxes])



def extract_rcrops(
    img: np.ndarray, polys: np.ndarray, dtype=np.float32, channels_last: bool = True
) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes
    Args:
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError("polys are expected to be quadrilateral, of shape (N, 4, 2)")

    # Project relative coordinates
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, :, 0] *= width
        _boxes[:, :, 1] *= height

    src_pts = _boxes[:, :3].astype(np.float32)
    # Preserve size
    d1 = np.linalg.norm(src_pts[:, 0] - src_pts[:, 1], axis=-1)
    d2 = np.linalg.norm(src_pts[:, 1] - src_pts[:, 2], axis=-1)
    # (N, 3, 2)
    dst_pts = np.zeros((_boxes.shape[0], 3, 2), dtype=dtype)
    dst_pts[:, 1, 0] = dst_pts[:, 2, 0] = d1 - 1
    dst_pts[:, 2, 1] = d2 - 1
    # Use a warp transformation to extract the crop
    crops = [
        cv2.warpAffine(
            img if channels_last else img.transpose(1, 2, 0),
            # Transformation matrix
            cv2.getAffineTransform(src_pts[idx], dst_pts[idx]),
            (int(d1[idx]), int(d2[idx])),
        )
        for idx in range(_boxes.shape[0])
    ]
    return crops


def generate_crops(
    pages: Union[List[np.ndarray], List[str]],
    loc_preds: List[np.ndarray],
    channels_last: bool,
    assume_straight_pages: bool = False,
) -> List[List[np.ndarray]]:

    extraction_fn = extract_crops if assume_straight_pages else extract_rcrops

    if all(isinstance(page, str) for page in pages):
        # i.e. it is textron 
        pages = [read_img_as_numpy(img) for img in pages]
        
    crops = [
        extraction_fn(page, _boxes[:, :4], channels_last=channels_last)  # type: ignore[operator]
        for page, _boxes in zip(pages, loc_preds)
    ]
    return crops


def prepare_crops_(
    pages: List[np.ndarray],
    loc_preds: List[np.ndarray],
    channels_last: bool,
    assume_straight_pages: bool = False,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:

    crops = generate_crops(pages, loc_preds, channels_last, assume_straight_pages)

    # Avoid sending zero-sized crops
    is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
    crops = [
        [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
        for page_crops, page_kept in zip(crops, is_kept)
    ]
    loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]

    return crops, loc_preds