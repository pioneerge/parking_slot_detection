import os
import numpy as np
import cv2
import argparse
import time

from tqdm import tqdm
from data_processing.KITTI_dataloader import KITTILoader
from utils.correspondece_constraint import *
from my_config import MyConfig as cfg


if cfg.network == 'vgg16':
    from model import vgg16 as nn
if cfg.network == 'mobilenet_v2':
    from model import mobilenet_v2 as nn


def predict(prediction_dir, label_dir, image_dir, calibration_file):
    # complie models
    model = nn.network()
    model.load_weights('3dbox_weights_mob.hdf5')
    # model.load_weights(args.w)

    # KITTI_train_gen = KITTILoader(subset='training')
    dims_avg, _ = KITTILoader(subset='tracklet').get_average_dimension()

    val_imgs = sorted([im for im in os.listdir(image_dir) if not im.startswith('.')])

    P2 = np.array([])
    for line in open(calibration_file):
        if 'P2' in line:
            P2 = line.split(' ')
            P2 = np.asarray([float(i) for i in P2[1:]])
            P2 = np.reshape(P2, (3, 4))

    for img in tqdm(val_imgs):
        image_file = os.path.join(image_dir, img)
        label_file = os.path.join(label_dir, img.replace('png', 'txt'))
        prediction_file = os.path.join(prediction_dir, img.replace('png', 'txt'))

        # write the prediction file
        with open(prediction_file, 'w') as predict:
            img = cv2.imread(image_file)
            img = np.array(img, dtype='float32')

            for line in open(label_file):
                line = line.strip().split(' ')
                obj = detectionInfo(line)
                xmin = int(obj.xmin)
                xmax = int(obj.xmax)
                ymin = int(obj.ymin)
                ymax = int(obj.ymax)
                if obj.name in cfg.KITTI_cat:

                    # cropped 2d bounding box
                    if xmin == xmax or ymin == ymax:
                        continue
                    # 2D detection area
                    patch = img[ymin : ymax, xmin : xmax]
                    try:
                        patch = cv2.resize(patch, (cfg.norm_h, cfg.norm_w))
                    except cv2.error:
                        continue
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
                    obj.alpha = recover_angle(bin_anchor, bin_confidence, cfg.bin)

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

    prediction_dir = cfg.labels3d
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    label_dir = cfg.labels
    image_dir = cfg.image_path
    calib_file = cfg.calib

    # make predictions
    predict(prediction_dir, label_dir, image_dir, calib_file)
