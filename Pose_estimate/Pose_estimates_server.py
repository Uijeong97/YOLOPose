# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import time
import cv2
import os
import csv
import sys

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize, makeMypose_df, resize_pose
from utils.pose_ftns import draw_body, get_people_pose, isStart, isInBox, draw_ground_truth, draw_truth
from algo.posture_dist import check_waist, check_knee, feedback_waist, check_ankle
from algo.speed_dist import check_speed
from sklearn.decomposition import PCA
from model import yolov3

from datetime import datetime

def estimatePose():
    parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
    # parser.add_argument("input_video", type=str,
    #                     help="The path of the input video.")
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/my_data/YOLOPose.names",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="./data/pose_weights/lunge_best",
                        help="The path of the weights to restore.")
    parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to save the video detection results.")
    args = parser.parse_args()

    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)

    color_table = get_color_table(args.num_class)

    # vid = cv2.VideoCapture(args.input_video)
    vid = cv2.VideoCapture('./data/demo/lunge_03.mp4')
    # vid = cv2.VideoCapture(r'C:\Users\soma\SMART_Referee\SMART_Referee_DL\data\lunge\video\lunge_03.mp4')
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    trainer_pose = pd.read_csv('./data/ground_truth/output_right.csv', header=None)
    trainer_pose = trainer_pose.loc[:, [0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
    pca_df = trainer_pose.loc[:, [1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
    pca_df.loc[:, [c for c in pca_df.columns if c % 2 == 1]] = pca_df.loc[:, [c for c in pca_df.columns if
                                                                              c % 2 == 1]] * video_width / 416
    pca_df.loc[:, [c for c in pca_df.columns if c % 2 == 0]] = pca_df.loc[:, [c for c in pca_df.columns if
                                                                              c % 2 == 0]] * video_height / 416
    pca_df = pca_df.astype(int)
    pca_df = pca_df.replace(0, np.nan)
    pca_df = pca_df.dropna()
    pca_df.describe()
    pca = PCA(n_components=1)
    pca.fit(pca_df)

    size = [video_width, video_height]
    list_p = []
    waist_err = 0
    critical_point = 0
    past_idx = 0
    startTrig = 0
    cntdown = 90
    t = 0
    TRLEN = len(trainer_pose)
    modify_ankle = pca_df.iloc[0, :].values
    base_rect = [(int(video_width / 4), int(video_height / 10)), (int(video_width * 3 / 4), int(video_height * 19 / 20))]
    c_knee = c_waist = c_speed = 0


    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()
            if args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            start_time = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            # rescale the coordinates to the original image
            if args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

            people_pose = get_people_pose(boxes_, labels_, base_rect)  # list-dict
            people_pose = np.array([p[1] for p in people_pose[0]]).flatten()  # dict-tuple -> list
            people_pose = people_pose[[0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]

            # Start Trigger
            if startTrig == 2:
                pass
            elif startTrig == 0:  # start
                # 기준 박스
                cv2.rectangle(img_ori, base_rect[0], base_rect[1], (0, 0, 255), 2)
                if isInBox(people_pose, base_rect[0], base_rect[1]):
                    # t_resize_pose = resize_pose(people_pose, trainer_pose.iloc[0, 1:].values)
                    t_resize_pose = resize_pose(people_pose, pca_df.iloc[0, :].values)
                    img_ori = draw_ground_truth(img_ori, t_resize_pose)
                    # img_ori = draw_ground_truth(img_ori, pca_df.iloc[0, :].values)
                    startTrig = isStart(people_pose, trainer_pose.iloc[0, 1:].values, size)

                    cv2.imshow('image', img_ori)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    print("박스안에 들어와주세요!!")
                    continue

            elif startTrig == 1:
                img_ori = draw_ground_truth(img_ori, pca_df.iloc[0, :].values)
                cv2.putText(img_ori, str(int(cntdown / 30)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
                cv2.imshow('image', img_ori)
                cntdown -= 1
                if cntdown == 0:
                    startTrig = 2
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue


            '''check ankle : 편차 40이상 발생시 전에 값 으로 업데이트'''
            people_pose = check_ankle(list_p, people_pose, modify_ankle, size)

            # f = open('user.csv', 'a', encoding='utf-8', newline='')
            # wr = csv.writer(f)
            # wr.writerow(people_pose)
            # ground truth 그리기

            list_p.append(people_pose)

            img_ori = draw_ground_truth(img_ori, pca_df.iloc[t, :].values)

            if check_waist(people_pose):
                waist_err += 1

            if waist_err is 60:  # waist_err는 60번 틀리면 피드백함
                feedback_waist()
                c_waist += 1
                waist_err = 0

            if trainer_pose.iloc[t, 0] == 1:  # t는 특정 시점 + i frame
                critical_point += 1
                if critical_point % 2 == 0:
                    my_pose = makeMypose_df(list_p)
                    c_speed = check_speed(my_pose, trainer_pose.iloc[past_idx: t + 1, 1:], pca, c_speed)
                    c_knee = check_knee(people_pose, c_knee)
                    modify_ankle = list_p[-1]
                    list_p = []
                    past_idx = t
            t += 1
            if t == TRLEN:
                break

            # img_ori = draw_body(img_ori, boxes_, labels_)
            # for i in range(len(boxes_)):
            #     x0, y0, x1, y1 = boxes_[i]
            #     plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])

            # 사용자 자세 그리기
            # img_ori = draw_truth(img_ori, people_pose)

            end_time = time.time()
            cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)

            cv2.imshow('image', img_ori)
            if args.save_video:
                videoWriter.write(img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        vid.release()
        cv2.destroyAllWindows()
        if args.save_video:
            videoWriter.release()

    f = open('./data/score/result.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    d = datetime.today().strftime("%Y/%m/%d")
    t = datetime.today().strftime("%H:%M:%S")
    wr.writerow([d,t, c_knee, c_waist, c_speed])