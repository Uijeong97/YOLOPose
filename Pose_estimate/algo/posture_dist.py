def check_waist(people_pose):
    Neck = people_pose[1]
    RHeap,LHeap = people_pose[8],people_pose[11]
    MHeap = ((RHeap[0]+LHeap[0])/2, (RHeap[1]+LHeap[1])/2)
    
    if MHeap[0]!=Neck[0]:
        return False
    else return True
        
def check_knee(people_pose):
    RKnee,RAnkle=people_pose[9],people_pose[10]
    LKnee,LAnkle=people_pose[12],people_pose[13]
    
    # compare x point
    ChKnee = RKnee if RKnee[0]<LKnee[0] else LKnee
    ChAnkle = RAnkle if RAnkle[0]<LAnkle[0] else LAnkle
    
    if ChKnee[0] < ChAnkle[0]: 
        feedback_knee()
    
def feedback_waist():
    print("허리와 엉덩이를 일직선으로 맞춰주세요")
    
def feedback_knee():
    print("다리를 수직으로 맞춰주세요")