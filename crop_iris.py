from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import exposure
from scipy.spatial.distance import euclidean
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import os

def extract_reflection(img, mask):
    """
    Extract highlights and pupil by removing pixels with high color value.
    [highlights, num_refl] = extract_reflection(img, mask)

    Parameters
    ----------
    img: list
        The image of the cornea.
    mask: list
        The mask of the iris (boolean).

    Returns
    -------
    highlights: list
        The mask of the highlights and pupil (boolean).
    num_refl: int
        The number of pixels from highlights.
    """
    negative_mask = np.logical_not(mask)
    roi_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    roi_V = roi_HSV[..., 2]
    roi_V = exposure.rescale_intensity(roi_V, in_range=(0, 255))
    roi_V[negative_mask] = 0
    highlights = roi_V >= 150
    num_refl = np.sum(highlights)

    pupil = roi_V <= 50
    pupil[negative_mask] = 0
    highlights = np.logical_or(highlights, pupil)

    return highlights, num_refl

def segment_iris(face_crop, eye_mask,radius_min_para,radius_max_para):
    """
    Crop iris according to the intersection of the cornea and the Hough circle.
    [img_ori_copy,img_copy, iris_mask, (cx_glob, cy_glob), radius_glob, (eye_cx, eye_cy), highlights_global, valid]
    = segment_iris(face_crop, eye_mask,radius_min_para,radius_max_para)

    Parameters
    ----------
    face_crop: list
        The features of the eye.
    eye_mask: list
        The mask of the cornea (boolean).
    radius_min_para: float
        A scale of the minimum radius of the Hough circle.
    radius_max_para: float
        A scale of the maximum radius of the Hough circle.

    Returns
    -------
    img_ori_copy: list
        The image of the eye with a white Hough circle.
    img_copy: list
        The image of the iris (the background is white).
    iris_mask: list
        The mask of the iris (boolean).
    (cx_glob, cy_glob): int
        The center of the best Hough circle.
    radius_glob: int
        The best radius of the the Hough circle.
    (eye_cx, eye_cy): int
        The mean of the coordinate of the cornea.
    highlights_global: list
        The mask of the highlights (boolean).
    valid: boolean
        A flag represents the circle exists or not.
    """
    img_copy = face_crop.copy()

    mask_coords = np.where(eye_mask != 0)
    mask_min_y = np.min(mask_coords[0])
    mask_max_y = np.max(mask_coords[0])
    mask_min_x = np.min(mask_coords[1])
    mask_max_x = np.max(mask_coords[1])

    roi_top = np.clip(mask_min_y, 0, face_crop.shape[0])
    roi_bottom = np.clip(mask_max_y, 0, face_crop.shape[0])
    roit_left = np.clip(mask_min_x, 0, face_crop.shape[1])
    roi_right = np.clip(mask_max_x, 0, face_crop.shape[1])

    roi_image = img_copy[roi_top:roi_bottom, roit_left:roi_right, :]

    roi_mask = eye_mask[roi_top:roi_bottom, roit_left:roi_right]

    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2LAB)
    roi_gray = roi_gray[..., 0]
    roi_gray = exposure.rescale_intensity(roi_gray, in_range=(0, 255))

    negative_mask = np.logical_not(roi_mask)
    roi_gray[negative_mask] = 0
    edges = canny(roi_gray, sigma=2.0, low_threshold=40, high_threshold=70)

    edges_mask = canny(roi_mask * 255, sigma=1.5, low_threshold=1, high_threshold=240)
    edges_mask = binary_erosion(edges_mask)
    edges_mask = binary_dilation(edges_mask)
    edges_mask = np.logical_not(edges_mask)

    # detect circles within radius range based on landmarks
    edges = np.logical_and(edges, edges_mask)
    diam = mask_max_x - mask_min_x
    radius_min = int(diam / radius_min_para)
    radius_max = int(diam / radius_max_para)
    hough_radii = np.arange(radius_min, radius_max, 1)
    hough_res = hough_circle(edges, hough_radii)
    # select best detection
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1, normalize=True)

    # select central point and diam/4 as fallback
    if radii is None or radii.size == 0:
        cx_glob = int(np.mean(mask_coords[1]))
        cy_glob = int(np.mean(mask_coords[0]))
        radius_glob = int(diam / 4.0)
        valid = False
    else:
        cx_glob = cx[0] + mask_min_x
        cy_glob = cy[0] + mask_min_y
        radius_glob = radii[0]
        valid = True

    # generate mask for iris
    iris_mask = np.zeros_like(eye_mask, dtype=np.uint8)
    cv2.circle(iris_mask, (cx_glob, cy_glob), radius_glob, 255, -1)
    img_ori_copy = face_crop.copy()
    cv2.circle(img_ori_copy, (cx_glob, cy_glob), radius_glob, (255, 255, 255), 1)  # img


    iris_mask = np.logical_and(iris_mask, eye_mask)

    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if iris_mask[i][j]==False:
                img_copy[i][j]=np.asarray([255,255,255])
    roi_iris = iris_mask[roi_top:roi_bottom, roit_left:roi_right]

    highlights, num_refl = extract_reflection(roi_image, roi_iris)
    highlights_global = np.zeros_like(eye_mask)
    highlights_coord = np.where(highlights != 0)
    highlights_coord[0][:] += mask_min_y
    highlights_coord[1][:] += mask_min_x
    highlights_global[highlights_coord] = 1

    eye_cx = int(np.mean(mask_coords[1]))
    eye_cy = int(np.mean(mask_coords[0]))
    return img_ori_copy,img_copy, iris_mask, (cx_glob, cy_glob), radius_glob, (eye_cx, eye_cy), highlights_global,num_refl, valid