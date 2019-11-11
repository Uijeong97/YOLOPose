import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.weightstats import ttest_ind

def check_speed(my_pose, tr_pose, pca):
    # trainer pose vs my pose 분포 비교
    my_pose = my_pose.values
    tr_pose = tr_pose.values
    my_dist = pca.fit_transform(my_pose)
    tr_dist = pca.fit_transform(tr_pose)

    v = ttest_ind(tr_dist, my_dist)[1]
    print(v, v> 0.05)
#     if slow : feedback_speed(slow)
#     elif fast : feedback_speed(fast)
#
# def feedback_speed(speed):
#     ;
#     if speed is slow:
#         print("")
#     else print("")