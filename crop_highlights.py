import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.measure import compare_ssim
import statistics
import heapq
import skimage.filters as filter
from skimage.filters import threshold_otsu, threshold_mean, thresholding


# def alignImages(im1, im2, mask_left_img, MAX_FEATURES = 50000, GOOD_MATCH_PERCENT = 1):
#     # Convert images to grayscale
#     im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#
#     # Detect ORB features and compute descriptors.
#     orb = cv2.ORB_create(MAX_FEATURES)
#     # orb = cv2.ORB_create()
#     keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
#
#     # Match features.
#     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#     # matcher = cv2.DescriptorMatcher_create(cv2.NORM_L2)
#     matches = matcher.match(descriptors1, descriptors2, None)
#
#     # Sort matches by score
#     matches.sort(key=lambda x: x.distance, reverse=False)
#
#     # Remove not so good matches
#     numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
#     matches = matches[:numGoodMatches]
#
#     # Draw top matches
#     imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#     # cv2.imwrite("matches.jpg", imMatches)
#
#     # Extract location of good matches
#     points1 = np.zeros((len(matches), 2), dtype=np.float32)
#     points2 = np.zeros((len(matches), 2), dtype=np.float32)
#
#     for i, match in enumerate(matches):
#         points1[i, :] = keypoints1[match.queryIdx].pt
#         points2[i, :] = keypoints2[match.trainIdx].pt
#
#     # Find homography
#     h, mask = cv2.estimateAffinePartial2D(points1, points2, method = cv2.RANSAC) ####2D
#     # h, mask = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC)  ####2D
#     # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)#####3D
#
#     # Use homography
#     h= np.float64([[1,0,h[0][2]],[0,1,h[1][2]]])
#     height, width, channels = im2.shape
#     im1Reg =cv2.warpAffine(im1, h, (width, height))#####2D
#     # im1Reg = cv2.warpPerspective(im1, h, (width, height))######3D
#     im1Reg_mask = cv2.warpAffine(mask_left_img, h, (width, height))
#     return im1Reg, h, im1Reg_mask

def matrix_reduce(iris_left_matrix, iris_right_matrix):
    """
    Shrink iris.
    [reduced_iris_left_matrix, reduced_iris_right_matrix] = matrix_reduce(iris_left_matrix, iris_right_matrix)

    Parameters
    ----------
    iris_left_matrix: list
        The mask of the left iris (boolean).
    iris_right_matrix: list
        The mask of the right iris (boolean).

    Returns
    -------
    reduced_iris_left_matrix: list
        The shrinking mask of the left iris (boolean).
    reduced_iris_right_matrix: list
        The shrinking mask of the right iris (boolean).
    """
    reduced_iris_left_matrix = np.zeros((iris_left_matrix.shape[0], iris_left_matrix.shape[1]), dtype=int)
    reduced_iris_right_matrix = np.zeros((iris_right_matrix.shape[0], iris_right_matrix.shape[1]), dtype=int)

    for i in range(iris_left_matrix.shape[0]):
        for j in range(iris_left_matrix.shape[1]):
            if iris_left_matrix[i][j]==1:
                if iris_left_matrix[i-1][j]==0 or \
                    iris_left_matrix[i+1][j]==0 or \
                    iris_left_matrix[i][j-1] == 0 or \
                    iris_left_matrix[i][j+1] == 0:
                    reduced_iris_left_matrix[i][j] = 0
                else:
                    reduced_iris_left_matrix[i][j] = 1
            else:
                reduced_iris_left_matrix[i][j] = 0

    for i in range(iris_right_matrix.shape[0]):
        for j in range(iris_right_matrix.shape[1]):
            if iris_right_matrix[i][j]==1:
                if iris_right_matrix[i-1][j]==0 or \
                    iris_right_matrix[i+1][j]==0 or \
                    iris_right_matrix[i][j-1] == 0 or \
                    iris_right_matrix[i][j+1] == 0:
                    reduced_iris_right_matrix[i][j] = 0
                else:
                    reduced_iris_right_matrix[i][j] = 1
            else:
                reduced_iris_right_matrix[i][j] = 0

    return reduced_iris_left_matrix, reduced_iris_right_matrix

def shiftbits(template, noshifts, matrix=False):
    """
    Shift the bit-wise highlight patterns.
    [templatenew] = shiftbits(template, noshifts, matrix=False)

    Parameters
    ----------
    template: list
        The mask of the highlights (boolean).
    noshifts: int
        The step size and direction of moving.
    matrix: bool
        Fill the empty item in the mask or not

    Returns
    -------
    templatenew: list
        The shifting mask of the highlights (boolean).
    """
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        if matrix:
            templatenew[:, x] = 0
        else:
            templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        if matrix:
            templatenew[:, x] = 0
        else:
            templatenew[:, x] = template[:, p + x]

    return templatenew

# def calculate_overlap(left_matrix,right_matrix):
#     sum = 0
#     for i in range(left_matrix.shape[0]):
#         for j in range(left_matrix.shape[1]):
#             if left_matrix[i][j]==1 and right_matrix[i][j]==1:
#                 sum=sum+1
#     return sum


def shift(img_left_matrix, img_right_matrix, negative_lr_shift, positive_lr_shift,negative_ud_shift, positive_ud_shift):
    """
    Move the mask of the left highlights up, down, left, and right to maximize the overlap with the mask of the right highlights to get the best IOU score.
    [opt_img_left_shift, max_overlap, opt_shift, IOU_score]
    = shift(img_left_matrix, img_right_matrix, negative_lr_shift, positive_lr_shift,negative_ud_shift, positive_ud_shift)

    Parameters
    ----------
    img_left_matrix: list
        The mask of the left highlights (boolean).
    img_right_matrix: list
        The mask of the right highlights (boolean).
    negative_lr_shift: int
        The maximum step size of moving left
    positive_lr_shift: int
        The maximum step size of moving right
    negative_ud_shift: int
        The maximum step size of moving up
    positive_ud_shift: int
        The maximum step size of moving down

    Returns
    -------
    opt_img_left_shift: list
        The optimal mask of the left highlights after moving.
    max_overlap: int
        The maximum number of overlapped pixels.
    opt_shift: int
        The optimal steps of moving
    IOU_score: float
        The best IoU score.
    """
    max_overlap = -math.inf
    IOU_score = 0
    opt_shift = []
    opt_img_left_shift = []
    for shifts_lr in range(negative_lr_shift, positive_lr_shift):
        for shift_ud in range(negative_ud_shift, positive_ud_shift):
            img_left_ud_shift = shiftbits(img_left_matrix, shift_ud, matrix =True)
            img_left_lr_shift = np.transpose(shiftbits(np.transpose(img_left_ud_shift), shifts_lr, matrix =True))
            m = np.sum(np.logical_and(img_left_lr_shift, img_right_matrix).astype(int))
            union_individual = np.sum(np.logical_or(img_left_lr_shift, img_right_matrix).astype(int))
            if m>=max_overlap:
                max_overlap = m
                if union_individual == 0:
                    IOU_score = 0
                else:
                    IOU_score = m / union_individual
                opt_shift = [shift_ud, shifts_lr]
                opt_img_left_shift = img_left_lr_shift
    return opt_img_left_shift, max_overlap, opt_shift, IOU_score

def process_aligned_image(iris_left, iris_right, iris_left_matrix, iris_right_matrix, l_highlights, r_highlights,
                          left_eye_image,right_eye_image, double_eye_img, double_eye_position_difference_list,
                          reduce = True, reduce_size = 2, threshold_scale_left =1, threshold_scale_right =1):
    """
    Crop highlights from the left and right iris.
    [iris_left, iris_right, left_recolor, right_recolor, left_recolor_resize, right_recolor_resize, IOU_score, double_eye_img_modified]
    = process_aligned_image(iris_left, iris_right, iris_left_matrix, iris_right_matrix, l_highlights, r_highlights,
                          left_eye_image,right_eye_image, double_eye_img, double_eye_position_difference_list,
                          reduce, reduce_size, threshold_scale_left, threshold_scale_right)

    Parameters
    ----------
    iris_left: list
        The image of the left iris (the background is white).
    iris_right: list
        The image of the right iris (the background is white).
    iris_left_matrix: list
        The mask of the left iris (boolean).
    iris_right_matrix: list
        The mask of the right iris (boolean).
    l_highlights: list
        The mask of the left highlights (boolean).
    r_highlights: list
        The mask of the right highlights (boolean).
    left_eye_image: list
        The features of the left eye.
    right_eye_image: list
        The features of the right eye.
    double_eye_img: list
        Consecutive double eyes area features taken from the face.
    double_eye_position_difference_list: ndarray
        The distance between new_eyes_position_list and double_eye_list.
    reduce: boolean
        Shrink iris or not.
    reduce_size: int
        The step size to shrink from the edge to the inside.
    threshold_scale_left: float
        Set a scale to increase or decrease the threshold for the left iris.
    threshold_scale_right: float
        Set a scale to increase or decrease the threshold for the right iris.

    Returns
    -------
    iris_left: list
        Resized image of the left iris.
    iris_right: int
        Resized image of the right iris.
    left_recolor: list
        Only show highlights (black color) in the left iris with the white background.
    right_recolor: int
        Only show highlights (black color) in the right iris with the white background.
    left_recolor_resize: list
        Resize the left iris image and show highlights with green color.
    right_recolor_resize: int
        Resize the right iris image and show highlights with red color.
    IOU_score: float
        Calculate IoU score based on the overlap of left highlights and right highlights.
    double_eye_img_modified: int
        Show highlights (green color in left and red color in right) on both eyes in the double_eye_img.
    """

    #####reduce iris boundary
    double_eye_img_modified = double_eye_img.copy()
    if reduce:
        for i in range(reduce_size):
            iris_left_matrix, iris_right_matrix = matrix_reduce(iris_left_matrix, iris_right_matrix)
        left_matrix = iris_left_matrix
        right_matrix = iris_right_matrix
        for i in range(left_matrix.shape[0]):
            for j in range(left_matrix.shape[1]):
                if left_matrix[i][j] != 1:
                    iris_left[i][j] = np.asarray([255, 255, 255])

        for i in range(right_matrix.shape[0]):
            for j in range(right_matrix.shape[1]):
                if right_matrix[i][j] != 1:
                    iris_right[i][j] = np.asarray([255, 255, 255])
    else:
        left_matrix = iris_left_matrix
        right_matrix = iris_right_matrix

    left_matrix_new = np.logical_xor(left_matrix, l_highlights)
    right_matrix_new = np.logical_xor(right_matrix, r_highlights)
    l_iris_vals = left_eye_image[left_matrix_new, :]
    r_iris_vals = right_eye_image[right_matrix_new, :]
    lIrisMean = np.mean(l_iris_vals, axis=0).astype(int)
    rIrisMean = np.mean(r_iris_vals, axis=0).astype(int)

    iris_left_ori_reduce_iris_color = iris_left.astype(int) - lIrisMean
    iris_right_ori_reduce_iris_color = iris_right.astype(int) - rIrisMean
    iris_left_ori_reduce_iris_color[iris_left_ori_reduce_iris_color < 0] = 0
    iris_right_ori_reduce_iris_color[iris_right_ori_reduce_iris_color < 0] = 0
    iris_left_ori_reduce_iris_color = iris_left_ori_reduce_iris_color.astype(np.uint8)
    iris_right_ori_reduce_iris_color = iris_right_ori_reduce_iris_color.astype(np.uint8)

    #### calculate threshold.
    # iris_left_Gray = cv2.cvtColor(iris_left_ori_reduce_iris_color, cv2.COLOR_BGR2GRAY)
    # iris_right_Gray = cv2.cvtColor(iris_right_ori_reduce_iris_color, cv2.COLOR_BGR2GRAY)
    iris_left_HSV = cv2.cvtColor(iris_left_ori_reduce_iris_color, cv2.COLOR_BGR2HSV)
    iris_right_HSV = cv2.cvtColor(iris_right_ori_reduce_iris_color, cv2.COLOR_BGR2HSV)

    left_color_list = []
    for i in range(left_matrix.shape[0]):
        for j in range(left_matrix.shape[1]):
            if left_matrix[i][j] == 1:
                left_color_list.append(iris_left_HSV[i][j])

    right_color_list = []
    for i in range(right_matrix.shape[0]):
        for j in range(right_matrix.shape[1]):
            if right_matrix[i][j] == 1:
                right_color_list.append(iris_right_HSV[i][j])

    # the_left_V = filter.threshold_otsu(np.asarray(left_color_list)[:, 2]) * threshold_scale_left
    # the_right_V = filter.threshold_otsu(np.asarray(right_color_list)[:, 2]) * threshold_scale_right
    the_left_V = filter.threshold_yen(np.asarray(left_color_list)[:, 2]) * threshold_scale_left
    the_right_V = filter.threshold_yen(np.asarray(right_color_list)[:, 2]) * threshold_scale_right

    #### extract highlights.
    left_recolor = np.zeros((iris_left.shape[0], iris_left.shape[1], 3), dtype=np.uint8)
    right_recolor = np.zeros((iris_right.shape[0], iris_right.shape[1], 3), dtype=np.uint8)
    left_recolor_matrix = np.zeros((iris_left.shape[0], iris_left.shape[1]), dtype=int)
    right_recolor_matrix = np.zeros((iris_right.shape[0], iris_right.shape[1]), dtype=int)

    for i in range(left_matrix.shape[0]):
        for j in range(left_matrix.shape[1]):
            if left_matrix[i][j]==1:
                if iris_left_HSV[i][j][2]>the_left_V:
                    left_recolor[i][j]=np.asarray([0, 0, 0])
                    left_recolor_matrix[i][j] = 1
                    double_eye_img_modified[i+double_eye_position_difference_list[0][1]][j+double_eye_position_difference_list[0][0]]=np.asarray([0, 255, 0])
                else:
                    left_recolor[i][j] = np.asarray([255, 255, 255])
                    left_recolor_matrix[i][j] = 0
            else:
                left_recolor[i][j] = np.asarray([255, 255, 255])
                left_recolor_matrix[i][j] = 0

    for i in range(right_matrix.shape[0]):
        for j in range(right_matrix.shape[1]):
            if right_matrix[i][j] == 1:
                if iris_right_HSV[i][j][2] > the_right_V:
                    right_recolor[i][j] = np.asarray([0, 0, 0])
                    right_recolor_matrix[i][j] = 1
                    double_eye_img_modified[i + double_eye_position_difference_list[1][1]][
                        j + double_eye_position_difference_list[1][0]] = np.asarray([255, 0, 0])
                else:
                    right_recolor[i][j] = np.asarray([255, 255, 255])
                    right_recolor_matrix[i][j] = 0
            else:
                right_recolor[i][j] = np.asarray([255, 255, 255])
                right_recolor_matrix[i][j] = 0

    #######Create 2 consistent images and matrix
    max_x_axis = max(iris_left.shape[0],iris_right.shape[0])
    max_y_axis = max(iris_left.shape[1], iris_right.shape[1])
    left_ori_resize = np.full((max_x_axis, max_y_axis, 3), 255, dtype=np.uint8)
    right_ori_resize = np.full((max_x_axis, max_y_axis, 3), 255, dtype=np.uint8)
    left_recolor_resize = np.full((max_x_axis, max_y_axis, 3), 255, dtype=np.uint8)
    right_recolor_resize = np.full((max_x_axis, max_y_axis, 3), 255, dtype=np.uint8)
    left_recolor_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    right_recolor_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    left_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)
    right_matrix_resize = np.zeros((max_x_axis, max_y_axis), dtype=int)

    for i in range(left_recolor.shape[0]):
        for j in range(left_recolor.shape[1]):
            # left_recolor_resize[i][j]=left_recolor[i][j]
            left_ori_resize[i][j] = iris_left[i][j]
            left_recolor_matrix_resize[i][j]=left_recolor_matrix[i][j]
            left_matrix_resize[i][j]= left_matrix[i][j]

    for i in range(right_recolor.shape[0]):
        for j in range(right_recolor.shape[1]):
            # right_recolor_resize[i][j] = right_recolor[i][j]
            right_ori_resize[i][j] = iris_right[i][j]
            right_recolor_matrix_resize[i][j] = right_recolor_matrix[i][j]
            right_matrix_resize[i][j] = right_matrix[i][j]


    ####do shift (or translation)
    opt_img_left_shift, max_overlap, opt_shift, IOU_score \
        = shift(left_recolor_matrix_resize, right_recolor_matrix_resize, \
                -int(max_x_axis/6), int(max_x_axis/6), -int(max_y_axis/5),int(max_y_axis/5))

    #####draw recolor resize
    left_matrix_ud_resize = shiftbits(left_matrix_resize, opt_shift[0], matrix=True)
    left_matrix_lr_resize = np.transpose(shiftbits(np.transpose(left_matrix_ud_resize), opt_shift[1], matrix=True))
    for i in range(opt_img_left_shift.shape[0]):
        for j in range(opt_img_left_shift.shape[1]):
            if left_matrix_lr_resize[i][j] == 1:
                if opt_img_left_shift[i][j] == 1:
                    left_recolor_resize[i][j] = np.asarray([0, 255, 0])
                else:
                    left_recolor_resize[i][j] = np.asarray([255, 255, 255])
            else:
                left_recolor_resize[i][j] = np.asarray([255,255,255])
            if right_matrix_resize[i][j]==1:
                if right_recolor_matrix_resize[i][j]==1:
                    right_recolor_resize[i][j] = np.asarray([255, 0, 0])
                else:
                    right_recolor_resize[i][j] = np.asarray([255, 255, 255])
            else:
                right_recolor_resize[i][j] = np.asarray([255,255,255])
    return iris_left, iris_right, left_recolor, right_recolor, left_recolor_resize, right_recolor_resize, IOU_score, double_eye_img_modified
