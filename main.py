import numpy as np
import argparse
import dlib
import logging
import cv2
import shutil
import sys
import os
import math
import argparse
import glob
from PIL import Image
from matplotlib import pyplot as plt
from crop_eyes import crop_eye, drawPoints, eye_detection
from crop_cornea import cornea_convex_hull
from crop_iris import  segment_iris
from crop_highlights import process_aligned_image


logging.basicConfig(level=logging.INFO,filename='buffalo_generated_visual_content_detection.log')
logger = logging.getLogger("buffalo_generated_visual_content_detection")



def Detection(args):
    #### 1. Read image
    try:
        data_name = os.path.splitext(os.path.basename(args.input))[0]
        # data_name = args.input.split("/")[-1]
        # data_name = data_name.split(".")[0]
    except:
        logger.error('The input image path or name is not correct. Please rename your image as name.type.')
        return False
        # exit()
    #### 2. Crop eye
    try:
        left_eye_image, right_eye_image, new_eyes_position_list, number_face, double_eye_img, double_eye_position_difference_list \
            = eye_detection(args.input, args.predictor_path)
    except:
        logger.error('Your image has some problems. It may not contain faces.')
        return False
        # exit()
    if number_face != 1:
        logger.error('Your image contains more than one face. However, our software can only work on one face.')
        return False
        # exit()
    #### 3. Crop cornea
    try:
        left_cornea, right_cornea, left_cornea_matrix, right_cornea_matrix \
            = cornea_convex_hull(left_eye_image, right_eye_image, new_eyes_position_list)
    except:
        logger.error('Crop cornea failed.')
        return False
        # exit()
    #### 4. Crop iris
    try:
        img_left, iris_left, l_iris, l_iris_center, l_radius, l_eye_center, l_highlights, l_num_refl, l_valid \
            = segment_iris(left_eye_image, left_cornea_matrix.astype(bool), args.radius_min_para,
                           args.radius_max_para)  # 'left'
        img_right, iris_right, r_iris, r_iris_center, r_radius, r_eye_center, r_highlights, r_num_refl, r_valid \
            = segment_iris(right_eye_image, right_cornea_matrix.astype(bool), args.radius_min_para,
                           args.radius_max_para)  # 'right'

        if l_num_refl==0 and r_num_refl==0:
            return False
    except:
        logger.error('Crop iris failed.')
        return False

        # exit()
    #### 5. Draw circles on double eyes
    try:
        double_eye_img_ori = double_eye_img.copy()
        new_left_eye = l_iris_center + double_eye_position_difference_list[0]
        new_right_eye = r_iris_center + double_eye_position_difference_list[1]
        cv2.circle(double_eye_img, (new_left_eye[0], new_left_eye[1]), l_radius, (0, 0, 255), 2)  # left
        cv2.circle(double_eye_img, (new_right_eye[0], new_right_eye[1]), r_radius, (0, 0, 255), 2)  # right
    except:
        logger.error('Draw circles on double eyes failed.')
        return False
        # exit()
    #### 6. Crop highlights
    try:
        iris_left_resize, iris_right_resize, left_recolor, right_recolor, \
        left_recolor_resize, right_recolor_resize, IOU_score, double_eye_img_modified \
            = process_aligned_image(iris_left, iris_right, l_iris, r_iris, l_highlights, r_highlights, left_eye_image,
                                    right_eye_image,
                                    double_eye_img, double_eye_position_difference_list, reduce=args.shrink,
                                    reduce_size=args.shrink_size, threshold_scale_left=args.threshold_scale_left,
                                    threshold_scale_right=args.threshold_scale_right)
    except:
        logger.error('Crop highlights failed.')
        return False
        # exit()

    #### 7. Save result
    try:
        ori_image = cv2.imread(args.input)
        ori_image = cv2.resize(ori_image, (double_eye_img_ori.shape[1], double_eye_img_ori.shape[1]))
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        space = np.full((2, double_eye_img_ori.shape[1], 3), 255, dtype=np.uint8)
        imgs_comb = np.vstack((ori_image, space, double_eye_img_ori, space, double_eye_img_modified))
        imgs_comb = Image.fromarray(imgs_comb)
        plt.imshow(imgs_comb)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("IoU:{}".format(f'{IOU_score:.4f}'))
        os.makedirs(args.output, exist_ok=True)
        plt.savefig('{}/{}_iris_final.png'.format(args.output, data_name), dpi=800, bbox_inches='tight',
                    pad_inches=0)
        plt.show()
        logger.info("IOU:{}".format(f'{IOU_score:.4f}'))
        logger.info("The result is saved in {}/{}_iris_final.png".format(args.output, data_name))
    except:
        logger.error('Save result failed.')
        return False

    return IOU_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./data/seed000000.png')    #seed000000 StyleGAN2_cars_car1134
    parser.add_argument('-output', type=str, default='./outputs')
    parser.add_argument('-radius_min_para', type=float, default=4.5) # radius_min = (mask_max_x - mask_min_x)/radius_min_para
    parser.add_argument('-radius_max_para', type=float, default=2.0) # radius_max = (mask_max_x - mask_min_x)/radius_max_para
    parser.add_argument('-shrink', type=bool, default=True)
    parser.add_argument('-shrink_size', type=int, default=2)
    parser.add_argument('-threshold_scale_left', type=float, default=1.2)
    parser.add_argument('-threshold_scale_right', type=float, default=1.2)
    parser.add_argument('-predictor_path', type=str, default='./shape_predictor/shape_predictor_68_face_landmarks.dat')
    args = parser.parse_args()
    IOU_score = Detection(args)
    print('IOU_score:', IOU_score)
    return IOU_score

if __name__ == "__main__":
    try:
        main()
    except:
        logger.error('Get IoU failed.')