import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageDraw


def draw_arm(img, pose_dict, line_color, side):
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())
    if side == "left":
        if 2 in ankle_li:
            draw.line((pose_dict[2][0], pose_dict[2][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=3)
            if 3 in ankle_li:
                draw.line((pose_dict[3][0], pose_dict[3][1], pose_dict[2][0], pose_dict[2][1]), fill=line_color,
                          width=3)
                if 4 in ankle_li:
                    draw.line((pose_dict[4][0], pose_dict[4][1], pose_dict[3][0], pose_dict[3][1]), fill=line_color,
                              width=3)
    elif side == "right":
        if 5 in ankle_li:
            draw.line((pose_dict[5][0], pose_dict[5][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color,
                      width=3)
            if 6 in ankle_li:
                draw.line((pose_dict[6][0], pose_dict[6][1], pose_dict[5][0], pose_dict[5][1]), fill=line_color,
                          width=3)
                if 7 in ankle_li:
                    draw.line((pose_dict[7][0], pose_dict[7][1], pose_dict[6][0], pose_dict[6][1]),
                              fill=line_color,
                              width=3)
    else:
        print("wrong input")
    return img

def draw_leg(img, pose_dict, line_color, side, r_foot_x, r_foot_y, r_foot_count , l_foot_x, l_foot_y, l_foot_count):
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())
    if side == "left":
        if 9 in ankle_li:
            draw.line((pose_dict[9][0], pose_dict[9][1], pose_dict[8][0], pose_dict[8][1]), fill=line_color, width=3)
            if 10 in ankle_li:
                draw.line((pose_dict[10][0], pose_dict[10][1], pose_dict[9][0], pose_dict[9][1]), fill=line_color,
                          width=3)
                if l_foot_count != 0:
                    draw.line((l_foot_x, l_foot_y, pose_dict[10][0], pose_dict[10][1]), fill=line_color,
                              width=3)

    elif side == "right":
        if 12 in ankle_li:
            draw.line((pose_dict[12][0], pose_dict[12][1], pose_dict[8][0], pose_dict[8][1]), fill=line_color, width=3)
            if 13 in ankle_li:
                draw.line((pose_dict[13][0], pose_dict[13][1], pose_dict[12][0], pose_dict[12][1]), fill=line_color,
                          width=3)
                if r_foot_count != 0:
                    draw.line((r_foot_x, r_foot_y, pose_dict[13][0], pose_dict[13][1]), fill=line_color,
                              width=3)

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
    line_color = (0,255,0)
    draw = ImageDraw.Draw(img)
    ankle_li = list(pose_dict.keys())
    if 1 not in ankle_li:
        return img
    else:
        # make head
        head_x, head_y, count = make_part((0, 15, 16, 17, 18), pose_dict)
        if count != 0:
            draw.line((head_x, head_y, pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=3)
            draw.ellipse((head_x - 8, head_y - 8, head_x + 8, head_y + 8), fill=(0,255,0))

        # left side
        img = draw_arm(img, pose_dict, line_color, side = "left")

        # right side
        img = draw_arm(img, pose_dict, line_color, side="right")

        # make r_foot
        r_foot_x, r_foot_y, r_foot_count = make_part((14,19,20,21), pose_dict)

        # make l_foot
        l_foot_x, l_foot_y, l_foot_count = make_part((11,22,23,24), pose_dict)

        # draw backbone
        if 8 not in ankle_li:
            # make center hip
            m_hip_x, m_hip_y, backbone_count = make_part((9, 12), pose_dict)
            if backbone_count != 0:
                draw.line((m_hip_x, m_hip_y, pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=3)

                if 9 in ankle_li:
                    draw.line((pose_dict[9][0], pose_dict[9][1], m_hip_x, m_hip_y), fill=line_color,
                              width=3)
                    if 10 in ankle_li:
                        draw.line((pose_dict[10][0], pose_dict[10][1], pose_dict[9][0], pose_dict[9][1]),
                                  fill=line_color,
                                  width=3)
                        if l_foot_count != 0:
                            draw.line((l_foot_x, l_foot_y, pose_dict[10][0], pose_dict[10][1]), fill=line_color,
                                      width=3)

                if 12 in ankle_li:
                    draw.line((pose_dict[12][0], pose_dict[12][1], m_hip_x, m_hip_y), fill=line_color,
                              width=3)
                    if 13 in ankle_li:
                        draw.line((pose_dict[13][0], pose_dict[13][1], pose_dict[12][0], pose_dict[12][1]),
                                  fill=line_color,
                                  width=3)
                        if r_foot_count != 0:
                            draw.line((r_foot_x, r_foot_y, pose_dict[10][0], pose_dict[10][1]), fill=line_color,
                                      width=3)


        else:
            draw.line((pose_dict[8][0], pose_dict[8][1], pose_dict[1][0], pose_dict[1][1]), fill=line_color, width=3)

            # left side
            img = draw_leg(img, pose_dict, line_color, "left", r_foot_x, r_foot_y, r_foot_count , l_foot_x, l_foot_y, l_foot_count)

            # right side
            img = draw_leg(img, pose_dict, line_color, "right", r_foot_x, r_foot_y, r_foot_count , l_foot_x, l_foot_y, l_foot_count)

        return img

def draw_body(img_root, boxes, labels):
    img = Image.open(img_root).convert('RGBA')
    draw = ImageDraw.Draw(img)

    labels = np.reshape(labels, (len(labels), 1))
    data = np.concatenate((boxes, labels), axis=1)

    people_li = data[data[:,4] == 27]

    pose_li = data[data[:,4] < 25]
    pose_data = np.zeros((len(pose_li), 3))
    pose_data[:,0] = (pose_li[:,0] + pose_li[:,2]) / 2
    pose_data[:,1] = (pose_li[:,1] + pose_li[:,3]) / 2
    pose_data[:,2] = pose_li[:,4]

    for i in range(len(people_li)):
        pose_dict = {}

        bound = 5
        body_li = pose_data[pose_data[:,0] > people_li[i,0] + bound]
        body_li = body_li[body_li[:, 1] > people_li[i, 1] + bound]
        body_li = body_li[body_li[:, 0] < people_li[i, 2] - bound]
        body_li = body_li[body_li[:, 1] < people_li[i, 3] - bound]

        for j in range(len(body_li)):
            if body_li[j,2] not in list(pose_dict.keys()):
                pose_dict[int(body_li[j,2])] = (int(body_li[j,0]), int(body_li[j,1]))

        ankle_li = list(pose_dict.keys())

        # draw point
        # for k in range(25):
        #     if k in ankle_li:
        #         r = 2
        #         draw.ellipse((pose_dict[k][0] - r, pose_dict[k][1] - r, pose_dict[k][0] + r, pose_dict[k][1] + r), fill=(255,0,0,0))

        img = line_body(img, pose_dict)

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img