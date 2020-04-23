import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from utils.read_dir import ReadDir
from data_processing.KITTI_dataloader import KITTILoader
from utils.correspondece_constraint import *
from data_processing.raw_data_processing import parse_raw_to_KITTI_form as to_kitti

import time

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2 as nn


def predict(prediction_dir, label_dir, image_dir, calib_dir):
    # complie models
    model = nn.network()
    model.load_weights('3dbox_weights_mob.hdf5')
    # model.load_weights(args.w)

    # KITTI_train_gen = KITTILoader(subset='training')
    dims_avg, _ = KITTILoader(subset='tracklet').get_average_dimension()

    val_imgs = sorted(os.listdir(image_dir))

    for img in tqdm(val_imgs):
        image_file = image_dir + img
        label_file = label_dir + img.replace('png', 'txt')
        prediction_file = prediction_dir + img.replace('png', 'txt')
        calibration_file = calib_dir + img.replace('png', 'txt')

        # write the prediction file
        with open(prediction_file, 'w') as predict:
            img = cv2.imread(image_file)
            img = np.array(img, dtype='float32')
            P2 = np.array([])
            for line in open(calibration_file):
                if 'P2' in line:
                    P2 = line.split(' ')
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3,4))

            for line in open(label_file):
                line = line.strip().split(' ')
                obj = detectionInfo(line)
                xmin = int(obj.xmin)
                xmax = int(obj.xmax)
                ymin = int(obj.ymin)
                ymax = int(obj.ymax)
                if obj.name in cfg().KITTI_cat:
                    # cropped 2d bounding box
                    if xmin == xmax or ymin == ymax:
                        continue
                    # 2D detection area
                    patch = img[ymin : ymax, xmin : xmax]
                    patch = cv2.resize(patch, (cfg().norm_h, cfg().norm_w))
                    # patch -= np.array([[[103.939, 116.779, 123.68]]])
                    patch /= 255.0
                    # extend it to match the training dimension
                    patch = np.expand_dims(patch, 0)

                    prediction = model.predict(patch)

                    dim = prediction[0][0]
                    bin_anchor = prediction[1][0]
                    bin_confidence = prediction[2][0]

                    # update with predict dimension
                    dims = dims_avg[obj.name] + dim
                    obj.h, obj.w, obj.l = np.array([round(dim, 2) for dim in dims])

                    # update with predicted alpha, [-pi, pi]
                    obj.alpha = recover_angle(bin_anchor, bin_confidence, cfg().bin)

                    # compute global and local orientation
                    obj.rot_global, rot_local = compute_orientaion(P2, obj)

                    # compute and update translation, (x, y, z)
                    obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

                    # output prediction label
                    output_line = obj.member_to_list()
                    output_line.append(1.0)
                    # Write regressed 3D dim and orientation to file
                    output_line = ' '.join([str(item) for item in output_line]) + '\n'
                    predict.write(output_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '-dir', type=str,
                        default='/Users/danilginzburg/Desktop/myimages', help='File to predict')
    args = parser.parse_args()

    prediction_dir = args.d + '_results'
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)
    label_dir = os.path.join(prediction_dir, '__labels')
    image_dir = args.d
    tracklet = 'kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync'
    calib_dir = os.path.join(prediction_dir, '__calib')

    # create label and calib files
    to_kitti.makedir(label_dir)
    to_kitti.makedir(calib_dir)

    transform, line_P2, P2 = to_kitti.read_transformation_matrix(tracklet)

    to_kitti.write_label(tracklet, label_dir, image_dir, transform, P2)
    to_kitti.write_calib(calib_dir, image_dir, line_P2)

    # make predictions
    predict(prediction_dir, label_dir, image_dir, calib_dir)
