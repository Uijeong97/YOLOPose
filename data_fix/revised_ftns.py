import json
import os
import numpy as np
import pandas as pd
import sys

def revised_people_label(image_dir, revised_label_dir):
    print("########################################")
    print("#                                      #")
    print("#                                      #")
    print("#              YOLOPose                #")
    print("#                                      #")
    print("#         Now Revising Data ...        #")
    print("#                                      #")
    print("#              End Soon                #")
    print("#                                      #")
    print("########################################")
    for root, dirs, files in os.walk(image_dir):
        for file in [f for f in files if os.path.splitext(f)[-1] == '.txt']:
            txt = open(os.path.join(root, file), 'r')
            while True:
                line = txt.readline()
                if not line: break
                line = line.rstrip('\n').split(' ')
                if int(line[0]) == 0:
                    sys.stdout = open(os.path.join(revised_label_dir, file), 'a')
                    print(
                        '%d %.6f %.6f %.6f %.6f' % (18, float(line[1]), float(line[2]), float(line[3]), float(line[4])))
            sys.stdout.close()


def concat_json_and_txt(json_dir, revised_label_dir):
    for root, dirs, files in os.walk(json_dir):
        for file in [f for f in files if os.path.splitext(f)[-1] == ".json"]:

            # read final label text file
            txt_name = file.replace("_keypoints.json", ".txt")
            sys.stdout = open(os.path.join(revised_label_dir, txt_name), 'a')

            # read json
            with open(os.path.join(root, file)) as json_file:
                json_data = json.load(json_file)
                df = pd.DataFrame(columns=["idx", "xmin", "ymin", "xmax", "ymax"])
                for people in json_data['people']:
                    keypoints = np.array(people['pose_keypoints_2d'])
                    keypoints = keypoints.reshape((len(keypoints) // 3, 3))

                    train_form = np.zeros((keypoints.shape[0], 4))
                    train_form[:, 0:2] = keypoints[:, :2]
                    train_form[:, 2:] = 10

                    people_df = pd.DataFrame(columns=['idx', 'x_center', 'y_center', 'width', 'hight'])
                    people_df['idx'] = np.array([str(i) for i in range(keypoints.shape[0])])
                    people_df[['x_center', 'y_center', 'width', 'hight']] = train_form / 416
                    people_df.drop(people_df.query('x_center == y_center == 0 ').index, inplace=True)

                    for i in people_df.values:
                        print("%s %.6f %.6f %.6f %.6f" % (i[0], i[1], i[2], i[3], i[4]))

            sys.stdout.close()