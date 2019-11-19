def check_waist(people_pose):
    Neck = people_pose[2:4]
    RHip,LHip = people_pose[4:6],people_pose[10:12]

    MHip = ((RHip[0]+LHip[0])/2, (RHip[1]+LHip[1])/2)

    if abs(MHip[0] - Neck[0]) >= 21.5:
        return True
    else:
        return False
        
def check_knee(people_pose, c_knee):
    RKnee,RAnkle = people_pose[6:8],people_pose[8:10]
    LKnee,LAnkle = people_pose[12:14],people_pose[14:]
    
    # compare x point
    ChKnee = RKnee if RKnee[0]>LKnee[0] else LKnee
    ChAnkle = RAnkle if RAnkle[0]>LAnkle[0] else LAnkle
    
    if ChKnee[0] - ChAnkle[0] > 2:
        c_knee += 1
        feedback_knee()
    return c_knee

def feedback_waist():
    print("허리와 엉덩이를 일직선으로 맞춰주세요")
    
def feedback_knee():
    print("다리를 수직으로 맞춰주세요")

def check_ankle(past_ankle, cur_ankle, trainer_ankle, size):
    threshold = min(size)/20

    if len(past_ankle) == 0:
        past_ankle = trainer_ankle
        diff = list(map(lambda x,y: abs(x-y), past_ankle, cur_ankle))
    else:
        past_ankle = past_ankle[-1]
        diff = list(map(lambda x,y: abs(x-y), past_ankle, cur_ankle))
    for i in range(len(diff)):
        if diff[i] > threshold:
            cur_ankle[i] = past_ankle[i]
    # diff = list(map(lambda x, y: abs(x - y), past_ankle, cur_ankle))
    # print(max(diff))

    return cur_ankle