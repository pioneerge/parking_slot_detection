'''
read the tracklets provided by kitti raw data
write the label file as kitti form
'''
import os
import cv2
import numpy as np
import shutil
from utils.read_dir import ReadDir

from xml.etree.ElementTree import ElementTree
import numpy as np
from warnings import warn
import itertools
from my_config import MyConfig as cfg

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0':STATE_UNSET, '1':STATE_INTERP, '2':STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1':OCC_UNSET, '0':OCC_VISIBLE, '1':OCC_PARTLY, '2':OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99':TRUNC_UNSET, '0':TRUNC_IN_IMAGE, '1':TRUNC_TRUNCATED, '2':TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
    """
    Representation an annotated object track

    Tracklets are created in function parseXML and can most conveniently used as follows:

    for trackletObj in parseXML(trackletFile):
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
      ... your code here ...
    #end: for all frames
    #end: for all tracklets

    absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
    amtOcclusion and amtBorders could be None

    You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int),
    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
    """

    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None   # n x 3 float array (x,y,z)
    rots = None    # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        """
        Creates Tracklet with no info set
        """
        self.size = np.nan*np.ones(3, dtype=float)

    def __str__(self):
        """
        Returns human-readable string representation of tracklet object

        called implicitly in
        print trackletObj
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

    def __iter__(self):
        """
        Returns an iterator that yields tuple of all the available data for each frame

        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amtOccs is None:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None), range(self.firstFrame, self.firstFrame+self.nFrames))
        else:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame+self.nFrames))
#end: class Tracklet


def parseXML(trackletFile):
    """
    Parses tracklet xml file and convert results to list of Tracklet objects

    :param trackletFile: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """

    # convert tracklet XML data to a tree structure
    eTree = ElementTree()
    print('Parsing tracklet file', trackletFile)
    with open(trackletFile) as f:
        eTree.parse(f)

    # now convert output to list of Tracklet objects
    trackletsElem = eTree.find('tracklets')
    tracklets = []
    trackletIdx = 0
    nTracklets = None
    for trackletElem in trackletsElem:
        #print 'track:', trackletElem.tag
        if trackletElem.tag == 'count':
            nTracklets = int(trackletElem.text)
            print('File contains', nTracklets, 'tracklets')
        elif trackletElem.tag == 'item_version':
            pass
        elif trackletElem.tag == 'item':
            #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
            # a tracklet
            newTrack = Tracklet()
            isFinished = False
            hasAmt = False
            frameIdx = None
            for info in trackletElem:
                #print 'trackInfo:', info.tag
                if isFinished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    newTrack.objectType = info.text
                elif info.tag == 'h':
                    newTrack.size[0] = float(info.text)
                elif info.tag == 'w':
                    newTrack.size[1] = float(info.text)
                elif info.tag == 'l':
                    newTrack.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    newTrack.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        #print 'trackInfoPose:', pose.tag
                        if pose.tag == 'count':     # this should come before the others
                            if newTrack.nFrames is not None:
                                raise ValueError('there are several pose lists for a single track!')
                            elif frameIdx is not None:
                                raise ValueError('?!')
                            newTrack.nFrames = int(pose.text)
                            newTrack.trans = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.rots = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.occs = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
                            newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
                            newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            frameIdx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frameIdx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                #print 'trackInfoPoseInfo:', poseInfo.tag
                                if poseInfo.tag == 'tx':
                                    newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                                    hasAmt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frameIdx += 1
                        else:
                            raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    isFinished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
            #end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not isFinished:
                warn('tracklet {0} was not finished!'.format(trackletIdx))
            if newTrack.nFrames is None:
                warn('tracklet {0} contains no information!'.format(trackletIdx))
            elif frameIdx != newTrack.nFrames:
                warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(trackletIdx, newTrack.nFrames, frameIdx))
            if np.abs(newTrack.rots[:,:2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amtOccs / amtBorders are not set, set them to None
            if not hasAmt:
                newTrack.amtOccs = None
                newTrack.amtBorders = None

            # add new tracklet to list
            tracklets.append(newTrack)
            trackletIdx += 1

        else:
            raise ValueError('unexpected tracklet info')
    #end: for tracklet list items

    print('Loaded', trackletIdx, 'tracklets.')

    # final consistency check
    if trackletIdx != nTracklets:
        warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets, trackletIdx))

    return tracklets
#end: function parseXML


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def obtain_2Dbox(dims, trans, rot, P2, img_xmax, img_ymax):
    '''
    obtain 2D bounding box based on 3D location values
    construct 3D bounding box at first, 2D bounding box is just the minimal and maximal values of 3D bounding box
    '''
    # generate 8 points for bounding box
    h, w, l = dims[0], dims[1], dims[2]
    tx, ty, tz = trans[0], trans[1], trans[2]

    R = np.array([[np.cos(rot), 0, np.sin(rot)],
                  [0, 1, 0],
                  [-np.sin(rot), 0, np.cos(rot)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([tx, ty, tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    for i in range(len(corners_2D[0, :])):
        if corners_2D[0, i] < 0:
            corners_2D[0, i] = 0
        elif corners_2D[0, i] > img_xmax:
            corners_2D[0, i] = img_xmax

    for j in range(len(corners_2D[1, :])):
        if corners_2D[1, j] < 0:
            corners_2D[1, j] = 0
        elif corners_2D[1, j] > img_ymax:
            corners_2D[1, j] = img_ymax

    xmin, xmax = int(min(corners_2D[0,:])), int(max(corners_2D[0,:]))
    ymin, ymax = int(min(corners_2D[1,:])), int(max(corners_2D[1,:]))

    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox


def local_ori(trans, rot):
    '''
    compute local orientation value based on global orientation and translation values
    '''
    local_ori = rot - np.arctan(trans[0]/trans[2])
    return round(local_ori,2)


def read_transformation_matrix(tracklet_path):
    for line in open(os.path.join(tracklet_path, 'calib_velo_to_cam.txt')):
        if 'R:' in line:
            R = line.strip().split(' ')
            R = np.asarray([float(number) for number in R[1:]])
            R = np.reshape(R, (3,3))

        if 'T:' in line:
            T = line.strip().split(' ')
            T = np.asarray([float(number) for number in T[1:]])
            T = np.reshape(T, (3,1))

    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'R_rect_00:' in line:
            R0_rect = line.strip().split(' ')
            R0_rect = np.asarray([float(number) for number in R0_rect[1:]])
            R0_rect = np.reshape(R0_rect, (3,3))

    # recifying rotation matrix
    R0_rect = np.append(R0_rect, np.zeros((3,1)), axis=1)
    R0_rect = np.append(R0_rect, np.zeros((1,4)), axis=0)
    R0_rect[-1,-1] = 1

    #The rigid body transformation from Velodyne coordinates to camera coordinates
    Tr_velo_to_cam = np.concatenate([R,T],axis=1)
    Tr_velo_to_cam = np.append(Tr_velo_to_cam, np.zeros((1,4)), axis=0)
    Tr_velo_to_cam[-1,-1] = 1

    transform = np.dot(R0_rect, Tr_velo_to_cam)

    # FIGURE OUT THE CALIBRATION
    for line in open(os.path.join(tracklet_path, 'calib_cam_to_cam.txt')):
        if 'P_rect_02' in line:
            line_P2 = line.replace('P_rect_02', 'P2')
            # print (line_P2)

    P2 = line_P2.split(' ')
    P2 = np.asarray([float(i) for i in P2[1:]])
    P2 = np.reshape(P2, (3,4))

    return transform, line_P2, P2


# Read the tracklets
def write_label(tracklet_path, label_path, image_path, transform, P2):
    for trackletObj in parseXML(os.path.join(tracklet_path, 'tracklet_labels.xml')):
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
            print(translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber)
            label_file = os.path.join(label_path, str(absoluteFrameNumber).zfill(10) + '.txt')
            image_file = os.path.join(image_path, str(absoluteFrameNumber).zfill(10) + '.png')
            img = cv2.imread(image_file)
            img_xmax, img_ymax = img.shape[1], img.shape[0]

            translation = np.append(translation, 1)
            translation = np.dot(transform, translation)
            translation = translation[:3]/translation[3]

            rot = -(rotation[2] + np.pi/2)
            if rot > np.pi:
                rot -= 2*np.pi
            elif rot < -np.pi:
                rot += 2*np.pi
            rot = round(rot, 2)

            local_rot = local_ori(translation, rot)

            bbox = obtain_2Dbox(trackletObj.size, translation, rot, P2, img_xmax, img_ymax)

            with open(label_file, 'a') as file_writer:
                line = [trackletObj.objectType] + [int(truncation),int(occlusion[0]),local_rot] + bbox + [round(size, 2) for size in trackletObj.size] \
                + [round(tran, 2) for tran in translation] + [rot]
                line = ' '.join([str(item) for item in line]) + '\n'
                file_writer.write(line)


def write_calib(calib_path, image_path, line_P2):
    for image in os.listdir(image_path):
        calib_file = calib_path + image.split('.')[0] + '.txt'

        # Create calib files
        with open(calib_file, 'w') as file_writer:
            file_writer.write(line_P2)


if __name__ == '__main__':
    base_dir = cfg().base_dir
    dir = ReadDir(base_dir=base_dir, subset='tracklet', tracklet_date='2011_09_26',
                  tracklet_file='2011_09_26_drive_0084_sync')
    tracklet_path = dir.tracklet_drive
    # label_path = dir.label_dir
    label_path = 'testlabel/'   # Label and Calib dirs are empty dirs. So we need only Image Dir and Prediction Dir
    image_path = dir.image_dir
    # calib_path = dir.calib_dir
    calib_path = 'testcalib/'

    makedir(label_path)
    makedir(calib_path)

    transform, line_P2, P2 = read_transformation_matrix(tracklet_path)

    write_label(tracklet_path, label_path, image_path, transform, P2)
    write_calib(calib_path, image_path, line_P2)
