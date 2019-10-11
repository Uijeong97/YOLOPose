# 동국대학교 빅데이터팜 결과전시회

﻿## Dongguk Univ BigData Farm

### 팀원

- 김성진

- 이태민

- 정의정

- 채진영

### Version

```
Python : 3.7
Tensorflow : 2.0.0
Opencv : 3.4.2
Pillow : 6.1.0
Pandas : 0.25.0
```

### Todo

- [x] Train COCO Human data
- [x] Make scoring pose algorithm
- [x] Test YOLOv3 with EfficientNet
- [x] Email to Stanford Univ Student to get dataset

### Weigths Download
The pretrained pose weights file can be downloaded [here](https://drive.google.com/open?id=1mBKWp90YHH-3pzIWzSWKovGqdzKOtNuj). Place this weights file under directory `./data/pose_weights/`

### Running Demo
```
python Pose_estimates.py
```
result:

![](https://github.com/comojin1994/YOLOPose/blob/master/Pose_estimate/detection_result.jpg?raw=true)

### Reference

- https://github.com/zzh8829/yolov3-tf2

- https://github.com/ZackPashkin/YOLOv3-EfficientNet-EffYolo
