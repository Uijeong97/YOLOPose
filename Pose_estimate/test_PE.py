# coding: utf-8

from __future__ import division, print_function

import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import argparse
import cv2
import os

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from utils.pose_ftns import draw_body, get_people_pose

from model import yolov3


parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--image_dir", type=str, default=r"./data/demo",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/YOLOPose.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/pose_weights/pose_best",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    for root, dirs, files in os.walk(args.image_dir):

        for file in [f for f in files if os.path.splitext(f)[-1] == '.jpg']:

            result = -1

            img_root = os.path.join(root, file)

            img_name = img_root.split('/')[-1]

            img_ori = cv2.imread(img_root)
            if args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            
            # rescale the coordinates to the original image
            if args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))
            

            people_pose=get_people_pose(boxes_, labels_) # list-dict
            print(people_pose[0])
            
            list_p=[]
            
            # dict-tuple -> list
            l = np.array([p for p in people_pose[0].values()]).flatten()
            print(l)
            list_p.append(l)
            print(list_p)
#             df=pd.DataFrame(data=np.array(list_p), columns=[]) # 30개마다
                
#             check_speed(f_queue_ps) if f_queue_ps.qsize() is T
            
            # normalize()
            # check_speed()
            # check_heap()
       
            # check critical point
            '''
            if i%T == CT
                check_knee(people_pose[0])  
            '''
            
            # draw body
            img_ori = draw_body(img_ori, people_pose)
            
            # draw yolo box
            # draw yolo box
#             for i in range(len(boxes_)):
#                  x0, y0, x1, y1 = boxes_[i]
#                  plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
                
#             cv2.imshow('Detection result', img_ori)
# #             cv2.imwrite('detection_result.jpg', img_ori)
#             cv2.waitKey(0)
