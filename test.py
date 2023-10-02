from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

# OpenCV need the 4 corners


def convertBack(x, y, w, h):  # OpenCV uses top left and bottom right
    xmin = int(round(x - (w / 2)))  # corner of rectangle box as input points
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# drawing bounding boxes on image from detections
# detections = detected objects, img= image of detected object, dim= dimensions of img
def DrawBBoxes(detections, img, dim, colors):
  # for "array of 6 elements" in detections --> label= detected class name, confidence= accuracy of detection, bbox= contains 4 elements x,y,w,h.
  i = 1
  # x and y is coordinates of center of bbox, w and h is width and height of bbox
  for label, confidence, bbox in detections:
   # \                             #Saving the values of in array bbox into x,y,w,h. x in 0 index position of array and so on...
   x, y, w, h = int((bbox[0]/dim)*width),
   int((bbox[1]/dim)*height), \
       int((bbox[2]/dim)*width), \
       int((bbox[3]/dim)*height)
   # calling function convertBack to get points for openCV
   xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
   pt1 = (xmin, ymin)
   pt2 = (xmax, ymax)
   # cv2.rectangle(image, start_point, end_point, color, thickness)
   cv2.rectangle(img, pt1, pt2, colors[label], 1)
   if label == 'with_mask':
     string = label + str(i)
     i += 1
   else:
     string = label
     cv2.putText(img, string + ":" + str(round(confidence, 2)),
                 (pt1[0]-5, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
     return img
###############################################################################################################


# define and load the trained models into GPU's"
invt_configPath = "darknet/cfg/yolov3_custom_train.cfg"
invt_weightPath = "darknet/backup/yolov3_custom_train_last.weights"
invt_metaPath = "darknet/data/yolo.data"


network, class_names, colors = darknet.load_network(
    invt_configPath,  invt_metaPath, invt_weightPath, batch_size=1)
# getting YOLO input image dimensions
invt_width = darknet.network_width(invt_network)
invt_height = darknet.network_height(invt_network)

# 'name'_detections= detected class label, frame=detected image, 416=dimension of YOLO detection , colors of labels
image = DrawBBoxes(invt_detections, frame, 416, colors)
# image = DrawBBoxes(ppe_detections,image,416)

# cv2.imshow("display window name", "video frame feed")
cv2.imshow('Inventory Detections', image)
cv2.waitKey(10)  # Displaying video for time in milliseconds per frame


cap.release()
# out.release()
cv2.destroyAllWindows()
###############################################################################################################
