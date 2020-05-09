import cv2


img = cv2.imread("/Users/danilginzburg/Projects/Project[S20]/3d-bounding-box-estimation-for-autonomous-driving/kitti_dataset/2011_09_26/2011_09_26_drive_0084_sync/data/0000000000.png")
bbox = [810, 184, 1242, 375]
bbox = [int(x) for x in bbox]
img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)