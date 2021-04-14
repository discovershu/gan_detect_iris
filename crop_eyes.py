import numpy as np
import dlib
from PIL import Image

def crop_eye(img, left, right):
    """
    Crop eyes from the face picture.
    [eyes_list, new_eyes_position_list, double_eye_list, double_eye_position_difference_list]
    = crop_eye(img, left, right)

    Parameters
    ----------
    img: list
        The features of the face.
    left: list
        The coordinates of the left eye.
    right: list
        The coordinates of the right eye.

    Returns
    -------
    eyes_list:
        The features of the left and right eyes.
    new_eyes_position_list: int
        It contains two sub-lists.
        The first sublist is the new position of the left eye.
        The second sublist is the new position of the right eye.
    double_eye_list: list
        Consecutive double eyes area features taken from the face.
    double_eye_position_difference_list: ndarray
        The distance between new_eyes_position_list and double_eye_list
    """
    eyes_list = []
    new_eyes_position_list = []
    rescale_position_list = []
    left_eye = left
    right_eye = right
    eyes = [left_eye, right_eye]
    lp_min = float("inf")
    rp_max = -float("inf")
    tp_min = float("inf")
    bp_max = -float("inf")
    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])
        if lp<lp_min:
            lp_min = lp
        if rp>rp_max:
            rp_max = rp
        if tp<tp_min:
            tp_min = tp
        if bp>bp_max:
            bp_max = bp
        w = rp - lp
        h = bp - tp
        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(img.shape[1], rp + 0.25 * w))#0.25
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(img.shape[0], bp + 1.75 * h))#1.75

        eyes_list.append(img[tp_:bp_, lp_:rp_, :])
        new_eye = eyes[j] - [lp_, tp_]
        new_eyes_position_list.append(new_eye)
        rescale_position_list.append([lp_, tp_])
    double_eye_list = img[tp_min-5:bp_max+5, lp_min-1:rp_max+1, :]
    double_eye_position_difference_list = np.asarray(rescale_position_list)-np.asarray([lp_min-1, tp_min-5])
    return eyes_list, new_eyes_position_list, double_eye_list, double_eye_position_difference_list

def drawPoints(faceLandmarks, startpoint, endpoint):
    """
    Get eye coordinates from face.
    [points] = drawPoints(faceLandmarks, startpoint, endpoint)

    Parameters
    ----------
    faceLandmarks:
        The landmarks/parts for the face.
    startpoint: int
        The start point of the eye.
    endpoint: int
        The end point of the eye.

    Returns
    -------
    points: list
        The coordinates of the eye based on faceLandmarks.
    """
    points = []
    for i in range(startpoint, endpoint+1):
        point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    return points


def eye_detection(data, predictor_path):
    """
    Give a face picture to detect human eyes.
    [left_eye_image, right_eye_image, new_eyes_position_list, len(dets), double_eye_list, double_eye_position_difference_list]
    =eye_detection(data, predictor_path)

    Parameters
    ----------
    data: str-type
        The path of the face picture.
    predictor_path: str-type
        The path of the dlib’s facial landmark predictor.

    Returns
    -------
    left_eye_image: list
        The features of the left eye.
    right_eye_image: list
        The features of the right eye.
    new_eyes_position_list: list
        It contains two sub-lists.
        The first sublist is the new position of the left eye.
        The second sublist is the new position of the right eye.
    len(dets): int
        The number of faces is detected in the picture.
    double_eye_list: list
        Consecutive double eyes area features taken from the face.
    double_eye_position_difference_list: ndarray
        The distance between new_eyes_position_list and double_eye_list
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img = dlib.load_rgb_image(data)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    d = dlib.rectangle(int(dets[0].left()), int(dets[0].top()),
                       int(dets[0].right()), int(dets[0].bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    left_eye = drawPoints(shape, 36, 41)
    right_eye = drawPoints(shape, 42, 47)

    eyes_list, new_eyes_position_list, double_eye_list, double_eye_position_difference_list = crop_eye(img, left_eye, right_eye)

    left_eye_image = eyes_list[0]
    right_eye_image = eyes_list[1]
    return left_eye_image, right_eye_image, new_eyes_position_list, len(dets), double_eye_list, double_eye_position_difference_list