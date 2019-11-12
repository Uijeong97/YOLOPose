import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.weightstats import ttest_ind, ztest

def check_speed(my_pose, tr_pose, pca, c_speed):
    # trainer pose vs my pose 분포 비교
    my_pose = my_pose.values
    tr_pose = tr_pose.values
    my_dist = pca.fit_transform(my_pose).flatten()
    tr_dist = pca.fit_transform(tr_pose).flatten()

    diff = np.array(list(map(lambda x,y : (x-y)**2, my_dist, tr_dist)))
    val = np.sqrt(diff.sum(axis=0))
    if val > 100:
        feedback_speed()
        c_speed += 1
    return c_speed

    # v = ttest_ind(tr_dist, my_dist, usevar='unequal')[1]
    # v = ztest(tr_dist, my_dist)
    # print(v, v< 0.05)
#     if slow : feedback_speed(slow)
#     elif fast : feedback_speed(fast)
#
def feedback_speed():
    print("운동속도를 맞춰 주세요")