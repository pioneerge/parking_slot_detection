import os


class MyConfig:

    # set the path to the base directory
    base_dir = '/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving'

    # set the path to the image's directory
    image_path = '/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync/image_02/data'

    # set the path to the weights file
    weights = '3dbox_weights_mob.hdf5'

    # set the base network: vgg16, vgg16_conv or mobilenet_v2
    network = 'vgg16'

    # set the bin size
    bin = 2

    # set the train/val split
    split = 0.8

    # set overlapping
    overlap = 0.1

    # set jittered
    jit = 3

    # set the normalized image size
    norm_w = 224
    norm_h = 224

    # set the batch size
    batch_size = 8

    KITTI_cat = ['Car']

    tracklet = os.path.join(base_dir, 'tracklet/')

    labels = os.path.join(base_dir, '2d_labels/')

    labels3d = os.path.join(base_dir, '3d_labels/')

    calib = os.path.join(base_dir, 'calib.txt')
