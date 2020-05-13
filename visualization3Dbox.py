import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

# from utils.read_dir import ReadDir
# from config import config as cfg
# from .. import config
from my_config import MyConfig as cfg
from utils.correspondece_constraint import *
from parking_space import find_parking_space

# car = ['car', 0.0, 0, 2.18, 318.0, 175.0, 484.0, 259.0, 1.6, 1.6, 3.78, -4.33, 2.62, 15.39, 3.9, 1.0]
# cfg = config.Config


def compute_distance(object1, object2):
    # object - list of 4 points
    distance = -1
    for point_obj1 in object1:
        for point_obj2 in object2:
            tmp = math.hypot(point_obj1[0] - point_obj2[0], point_obj1[1] - point_obj2[1])
            if distance == -1:
                distance = tmp
            distance = tmp if distance > tmp else distance
    return distance


def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    # h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -l / 2

    # print(x_corners, "------------" ,z_corners)

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])

    # print("\n", corners_2D)

    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))


def draw_birdeyes(ax2, line_p, shape, line_gt=None, color='green'):
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)

    # default_car = compute_birdviewbox(car, shape, scale)
    if line_gt is not None:
        gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)
        codes = [Path.LINETO] * gt_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(gt_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
        ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color=color, label='prediction')
    ax2.add_patch(p)

    return pred_corners_2d


def draw_free_slots(ax2, free_slots, color='orange'):
    for free_slot in free_slots:
        codes = [Path.LINETO] * free_slot.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(free_slot, codes)
        p = patches.PathPatch(pth, fill=False, color=color, label='free space')
        ax2.add_patch(p)


def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D


def draw_3Dbox(ax, P2, line, color):

    corners_2D = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)


def visualization(args, image_path, label_path, calibration_file, pred_path,
                  dataset, VEHICLES):
    for line in open(calibration_file):
        if 'P2' in line:
            P2 = line.split(' ')
            P2 = np.asarray([float(i) for i in P2[1:]])
            P2 = np.reshape(P2, (3, 4))

    for index in range(len(dataset)):
        image_file = os.path.join(image_path, dataset[index]+ '.png')
        label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')

        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        # fig.tight_layout()
        gs = GridSpec(1, 4)
        gs.update(wspace=0)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])

        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
        image = Image.open(image_file).convert('RGB')
        shape = 900
        birdimage = np.zeros((shape, shape, 3), np.uint8)
        cars = list()

        with open(label_file) as f1, open(prediction_file) as f2:
            for line_gt, line_p in zip(f1, f2):
                line_gt = line_gt.strip().split(' ')
                line_p = line_p.strip().split(' ')

                truncated = np.abs(float(line_p[1]))
                occluded = np.abs(float(line_p[2]))
                trunc_level = 1 if args.a == 'training' else 255

                # truncated object in dataset is not observable
                if line_p[0] in VEHICLES and truncated < trunc_level:
                    color = 'green'
                    draw_3Dbox(ax, P2, line_p, color)
                    car = draw_birdeyes(ax2, line_p, shape, color=color)
                    cars.append(car)

        # determine parking space
        free_slots = find_parking_space(cars, shape)
        print(free_slots)
        draw_free_slots(ax2, free_slots)

        ax.imshow(image)
        ax.set_xticks([])  # remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

        # visualize bird eye view
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # add legend
        handles, labels = ax2.get_legend_handles_labels()
        print("Handles:  ", handles[-1])
        print("Labels:  ", labels[-1])
        legend = ax2.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='lower right',
                            fontsize='x-small', framealpha=0.2)
        for text in legend.get_texts():
            print("TEXT:   ", text)
            plt.setp(text, color='w')


        print(os.path.join(args.path, dataset[index]))
        # print(dataset[index])
        if args.save == False:
            plt.show()
        else:
            fig.savefig(os.path.join(args.path, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        # video_writer.write(np.uint8(fig))


def main(args):
    base_dir = '/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset'
    # dir = ReadDir(base_dir=base_dir, subsetset=args.a,
    #               tracklet_date='2011_09_26', tracklet_file='2011_09_26_drive_0084_sync')
    label_path = args.l
    image_path = args.d
    calib_path = '/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/calib.txt'
    pred_path = args.pred

    dataset = [name.split('.')[0] for name in sorted(os.listdir(image_path)) if not name.startswith('.')]

    VEHICLES = cfg().KITTI_cat

    visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default='./result1', help='Output Image folder')

    parser.add_argument('-d', '-dir', type=str,
                        default='/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync/data', help='File to predict')
    parser.add_argument('-l', '-label', default='/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync/testlabel', type=str)
    parser.add_argument('-pred', '-prediction', default='/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync/data_results', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    main(args)
