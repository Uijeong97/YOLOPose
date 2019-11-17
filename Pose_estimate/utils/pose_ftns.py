import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageDraw

def isInBox(people_pose, box_1, box_2):
    evenidx = [0, 2, 4, 6, 8, 10, 12, 14]
    oddidx = [1, 3, 5, 7, 8, 11, 13, 15]
    T_p = (len([True for d in people_pose[evenidx] if box_1[0] < d & d < box_2[0]])+\
    len([True if box_1[1] < d & d < box_2[1] else False for d in people_pose[oddidx]]))/16

    if T_p >= 0.8:
        return True
    else:
        return False

def isStart(people_pose, trainer_pose, size):

    # if people_pose
    mse = ((people_pose - trainer_pose)**2).mean(axis = 0)
    mse = np.sqrt(mse)

    t = min(size)/40
    if mse > t:
        print("자세를 맞춰주세요!!")
        return 0
    else:
        return 1
        
    
def get_people_pose(boxes, labels, base_rect):
    labels = np.reshape(labels, (len(labels), 1))
    data = np.concatenate((boxes, labels), axis=1)
    pose_li = data[data[:,4] < 18]
    pose_data = np.zeros((len(pose_li), 3))
    pose_data[:,0] = (pose_li[:,0] + pose_li[:,2]) / 2
    pose_data[:,1] = (pose_li[:,1] + pose_li[:,3]) / 2
    pose_data[:,2] = pose_li[:,4]

    people_body_li=[]

    pose_dict = {}

    body_li = pose_data[pose_data[:, 0] > base_rect[0][0]]
    body_li = body_li[body_li[:, 1] > base_rect[0][1]]
    body_li = body_li[body_li[:, 0] < base_rect[1][0]]
    body_li = body_li[body_li[:, 1] < base_rect[1][1]]

    for j in range(len(body_li)):
        if body_li[j,2] not in list(pose_dict.keys()):
            pose_dict[int(body_li[j,2])] = (int(body_li[j,0]), int(body_li[j,1]))
    l_li=[i for i in range(18) if i not in list(pose_dict.keys())]
    for l_item in l_li:
        pose_dict[l_item] = (0,0)
    pose_dict = sorted(pose_dict.items())
    people_body_li.append(pose_dict)

    return people_body_li


def get_multi_people_pose(boxes, labels):
    labels = np.reshape(labels, (len(labels), 1))
    data = np.concatenate((boxes, labels), axis=1)
    people_li = data[data[:, 4] == 18]
    pose_li = data[data[:, 4] < 18]
    pose_data = np.zeros((len(pose_li), 3))
    pose_data[:, 0] = (pose_li[:, 0] + pose_li[:, 2]) / 2
    pose_data[:, 1] = (pose_li[:, 1] + pose_li[:, 3]) / 2
    pose_data[:, 2] = pose_li[:, 4]

    people_body_li = []
    for i in range(len(people_li)):
        pose_dict = {}

        bound = 5
        body_li = pose_data[pose_data[:, 0] > people_li[i, 0] + bound]
        body_li = body_li[body_li[:, 1] > people_li[i, 1] + bound]
        body_li = body_li[body_li[:, 0] < people_li[i, 2] - bound]
        body_li = body_li[body_li[:, 1] < people_li[i, 3] - bound]

        for j in range(len(body_li)):
            if body_li[j, 2] not in list(pose_dict.keys()):
                pose_dict[int(body_li[j, 2])] = (int(body_li[j, 0]), int(body_li[j, 1]))
        l_li = [i for i in range(18) if i not in list(pose_dict.keys())]
        for l_item in l_li:
            pose_dict[l_item] = (0, 0)
        pose_dict = sorted(pose_dict.items())
        people_body_li.append(pose_dict)

    return people_body_li

def draw_arm(img, pose_dict, line_color, side, w):
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())
    if side == "left":
        if 2 in ankle_li:
            draw.line((pose_dict[2][0], pose_dict[2][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=w)
            if 3 in ankle_li:
                draw.line((pose_dict[3][0], pose_dict[3][1], pose_dict[2][0], pose_dict[2][1]), fill=line_color,
                          width=w)
                if 4 in ankle_li:
                    draw.line((pose_dict[4][0], pose_dict[4][1], pose_dict[3][0], pose_dict[3][1]), fill=line_color,
                              width=w)
    elif side == "right":
        if 5 in ankle_li:
            draw.line((pose_dict[5][0], pose_dict[5][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color,
                      width=w)
            if 6 in ankle_li:
                draw.line((pose_dict[6][0], pose_dict[6][1], pose_dict[5][0], pose_dict[5][1]), fill=line_color,
                          width=w)
                if 7 in ankle_li:
                    draw.line((pose_dict[7][0], pose_dict[7][1], pose_dict[6][0], pose_dict[6][1]),
                              fill=line_color,
                              width=w)
    else:
        print("wrong input")
    return img

def draw_leg(img, pose_dict, line_color, side, w):
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())
    if side == "left":
        if 9 in ankle_li:
            draw.line((pose_dict[9][0], pose_dict[9][1], pose_dict[8][0], pose_dict[8][1]), fill=line_color, width=w)
            if 10 in ankle_li:
                draw.line((pose_dict[10][0], pose_dict[10][1], pose_dict[9][0], pose_dict[9][1]), fill=line_color,
                          width=w)

    elif side == "right":
        if 12 in ankle_li:
            draw.line((pose_dict[12][0], pose_dict[12][1], pose_dict[11][0], pose_dict[11][1]), fill=line_color, width=w)
            if 13 in ankle_li:
                draw.line((pose_dict[13][0], pose_dict[13][1], pose_dict[12][0], pose_dict[12][1]), fill=line_color,
                          width=w)

    else:
        print("wrong input")

    return img

def make_part(set, pose_dict):
    ankle_li = list(pose_dict.keys())
    count = 0
    x, y = 0, 0
    for i in set:
        if i in ankle_li:
            x += pose_dict[i][0]
            y += pose_dict[i][1]
            count += 1
    if count != 0:
        x /= count
        y /= count
    return x, y, count

def line_body(img, pose_dict):
    line_color = (145,56,40)
    w = 5
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())

    if 1 not in ankle_li:
        return img
    else:
        # make head
        head_x, head_y, count = make_part((0, 14, 15, 16, 17), pose_dict)
        if count != 0:
            r = 20
            draw.line((head_x, head_y, pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=w)
            draw.ellipse((head_x - r, head_y - r, head_x + r, head_y + r), fill=line_color)
        if 2 in ankle_li:
            # left side
            img = draw_arm(img, pose_dict, line_color, "left", w)

        if 5 in ankle_li:
            # right side
            img = draw_arm(img, pose_dict, line_color, "right", w)

        if 8 in ankle_li:
            draw.line((pose_dict[8][0], pose_dict[8][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=w)

            # left side
            img = draw_leg(img, pose_dict, line_color, "left", w)

        if 11 in ankle_li:
            draw.line((pose_dict[11][0], pose_dict[11][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=w)

            # right side
            img = draw_leg(img, pose_dict, line_color, "right", w)

        return img

def draw_body(img_root, people_pose):
    img = Image.fromarray(img_root)
    draw = ImageDraw.Draw(img)

        # draw point
        # for k in range(25):
        #     if k in ankle_li:
        #         r = 2
        #         draw.ellipse((pose_dict[k][0] - r, pose_dict[k][1] - r, pose_dict[k][0] + r, pose_dict[k][1] + r), fill=(255,0,0,0))
    
    for pose_dict in people_pose:
        img = line_body(img, pose_dict)
        
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = np.array(img)

    return img

def draw_ground_truth(img_ori, trainer_pose):
    img = Image.fromarray(img_ori)
    draw = ImageDraw.Draw(img)
    line_color = (145, 56, 40)
    w = 5; r=20

    draw.ellipse((trainer_pose[0] - r, trainer_pose[1] - r, trainer_pose[0] + r, trainer_pose[1] + r), fill=line_color)
    draw.line((trainer_pose[0], trainer_pose[1], trainer_pose[2], trainer_pose[3]), fill=line_color, width=w)
    draw.line((trainer_pose[2], trainer_pose[3], trainer_pose[4], trainer_pose[5]), fill=line_color, width=w)
    draw.line((trainer_pose[4], trainer_pose[5], trainer_pose[6], trainer_pose[7]), fill=line_color, width=w)
    draw.line((trainer_pose[6], trainer_pose[7], trainer_pose[8], trainer_pose[9]), fill=line_color, width=w)
    draw.line((trainer_pose[2], trainer_pose[3], trainer_pose[10], trainer_pose[11]), fill=line_color, width=w)
    draw.line((trainer_pose[10], trainer_pose[11], trainer_pose[12], trainer_pose[13]), fill=line_color, width=w)
    draw.line((trainer_pose[12], trainer_pose[13], trainer_pose[14], trainer_pose[15]), fill=line_color, width=w)

    img = np.array(img)

    return img

def draw_truth(img_ori, trainer_pose):
    img = Image.fromarray(img_ori)
    draw = ImageDraw.Draw(img)
    line_color = (0, 0, 255)
    w = 5; r= 20

    draw.ellipse((trainer_pose[0] - r, trainer_pose[1] - r, trainer_pose[0] + r, trainer_pose[1] + r), fill=line_color)
    draw.line((trainer_pose[0], trainer_pose[1], trainer_pose[2], trainer_pose[3]), fill=line_color, width=w)
    draw.line((trainer_pose[2], trainer_pose[3], trainer_pose[4], trainer_pose[5]), fill=line_color, width=w)
    draw.line((trainer_pose[4], trainer_pose[5], trainer_pose[6], trainer_pose[7]), fill=line_color, width=w)
    draw.line((trainer_pose[6], trainer_pose[7], trainer_pose[8], trainer_pose[9]), fill=line_color, width=w)
    draw.line((trainer_pose[2], trainer_pose[3], trainer_pose[10], trainer_pose[11]), fill=line_color, width=w)
    draw.line((trainer_pose[10], trainer_pose[11], trainer_pose[12], trainer_pose[13]), fill=line_color, width=w)
    draw.line((trainer_pose[12], trainer_pose[13], trainer_pose[14], trainer_pose[15]), fill=line_color, width=w)

    img = np.array(img)

    return img