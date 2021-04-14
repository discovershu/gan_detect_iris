import numpy as np
import cv2

def cornea_convex_hull(left_eye_img, right_eye_img, new_eyes_position_list):
    """
    Crop corneas from the left and right eyes.
    [left_cornea, right_cornea, left_cornea_matrix, right_cornea_matrix]
    = cornea_convex_hull(left_eye_img, right_eye_img, new_eyes_position_list)

    Parameters
    ----------
    left_eye_img: list
        The features of the left eye.
    right_eye_img: list
        The features of the right eye.
    new_eyes_position_list: list
        It contains two sub-lists.
        The first sublist is the new position of the left eye.
        The second sublist is the new position of the right eye.

    Returns
    -------
    left_cornea: ndarray
        The features of the left cornea (gray image).
    right_cornea: ndarray
        The features of the right cornea (gray image).
    left_cornea_matrix: ndarray
        The mask of the left cornea (binary).
    right_cornea_matrix: ndarray
        The mask of the right cornea (binary).
    """
    left_cornea=np.zeros((left_eye_img.shape[0], left_eye_img.shape[1], 3), np.uint8)
    left_cornea_matrix = np.zeros((left_eye_img.shape[0], left_eye_img.shape[1]), np.uint8)
    cv2.fillConvexPoly(left_cornea, new_eyes_position_list[0], (255, 255, 255))

    right_cornea=np.zeros((right_eye_img.shape[0], right_eye_img.shape[1], 3), np.uint8)
    right_cornea_matrix = np.zeros((right_eye_img.shape[0], right_eye_img.shape[1]), np.uint8)
    cv2.fillConvexPoly(right_cornea, new_eyes_position_list[1], (255, 255, 255))

    left_cornea_img=left_cornea
    right_cornea_img=right_cornea

    for i in range(left_cornea_img.shape[0]):
        for j in range(left_cornea_img.shape[1]):
            if left_cornea_img[i][j][0] == 255 and left_cornea_img[i][j][1] == 255 and left_cornea_img[i][j][2] == 255:
                left_cornea_matrix[i][j]=1

    for i in range(right_cornea_img.shape[0]):
        for j in range(right_cornea_img.shape[1]):
            if right_cornea_img[i][j][0] == 255 and right_cornea_img[i][j][1] == 255 and right_cornea_img[i][j][2] == 255:
                right_cornea_matrix[i][j]=1
    return left_cornea, right_cornea, left_cornea_matrix, right_cornea_matrix