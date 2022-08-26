import cv2
import numpy as np
import os

from Detector import Shapes, Trackers
from SiameseNet import SiameseNetwork
from LandmarkDataset import getDataset

import torch.nn.functional as F
import torch
from math import atan,pi
import sys
import json
from datetime import datetime
import time
import csv



try:
    os.mkdir('RecallResults')
except FileExistsError:
    pass

SAVENAME="RecallResults/"+str(datetime.now()).replace(' ','_').replace(':','_').replace('.','_')+'_Recall.mkv'  


args = sys.argv
if not len(args)>=4:
    print("Usage: \n \t python3 {} [Model Name] [epoch] [input] [env]".format(args[0]))
    exit()
NAME = args[1]+'_'+args[2]+'.pth'
IN = args[3]
env = args[4]
epoch = int(args[2])

logCSV = open('RecogLog_'+env+'.csv','a')
logWriter = csv.writer(logCSV)
# NAME = "SiamNetModels/"+NAME

def distance(o1,o2):
    euclidean_distance = F.pairwise_distance(o1, o2)
    # print(euclidean_distance)
    d = 1 - atan(euclidean_distance.item())/(pi/2)
    return d


print("Loading Network")
net = SiameseNetwork().cuda()
net.load_state_dict(torch.load(NAME))
print(net.eval())

print("Loading Landmarks")
ds = getDataset('landmarks')

landmarks = ds.getLandmarks()
recallable_ids = list(landmarks.keys())
print("Recallable Ids:{}".format(recallable_ids))

centroid_info = json.load(open("CentroidInfo.json"))


vid = cv2.VideoCapture(IN)
detector = Shapes()
trackers = Trackers(5)
frame_number=1
last_known_match ={}
prev_frame_time=time.time()
current_frame_time=0
fps=0
fpss=[]
pvals = []
highest_seen = None

while(vid.isOpened()):
    ret,image = vid.read()
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if not ret:
        break

    rois=detector.contours(image)
    tracked = trackers.updateLandmarks(image)
    ids = list(tracked.keys())
    seen_ids = []
    for id in ids:
        if not tracked[id]==None:
            seen_ids.append(id)

    
    
    for id in seen_ids:
        # if tracked[id]==None:
        #     continue

        x,y,w,h = tracked[id]
        candidate = image[int(y):int(y+h),int(x):int(x+w)]
        try:
            candidate = cv2.resize(candidate,(64,64)).T
            candidate = torch.Tensor(candidate)[None, ...].cuda()
        except cv2.error as e:
            # print(e)
            continue

        matches = {}
        for _id in landmarks:
            o1,o2 = net(candidate,landmarks[_id])
            p = distance(o1,o2)
            print("Id:{} P:{}".format(_id,p))
            if(p<0.5):
                matches[_id]=0
            else:
                matches[_id]=p
        best_match = max(matches, key=matches.get)
        if matches[best_match]==0:
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)),(255,255,0) , 4)
            # cv2.putText(image,"{:.2f}".format(matches[best_match]), (int(x)+int(w),int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

            try:
                cv2.putText(image,str(last_known_match[id]), (int(x)+int(w/2),int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),3)
            except KeyError:
                pass

        else:
            last_known_match[id]=best_match
            pvals.append(matches[best_match])
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)),(255,0,255) , 4)
            cv2.putText(image,"{:.2f}".format(matches[best_match]), (int(x)+int(w),int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)
            cv2.putText(image,str(best_match), (int(x)+int(w/2),int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),3)

        print(last_known_match)
        try:
            highest_seen = max(list(map(int,last_known_match.values())))

            target = min([int(id) for id in recallable_ids if int(id)>highest_seen])
        except ValueError:
            target=min([int(id) for id in recallable_ids])
        

        try:
            direction = np.sign(centroid_info[str(target)][0][0]-centroid_info[str(highest_seen)][1][0])*np.sign(centroid_info[str(target)][1][0])
        except KeyError:
            print("Target:{},Current:{}".format(target,highest_seen))
            direction=0
        print("Highest Found:{},Next Target:{},Direction:{}".format(highest_seen,target,direction))
        point=" "
        point_pos = (int(image.shape[1]/3),int(image.shape[0]/2))

        if direction<0:
            point="<<{}".format(target)
            point_pos = (int(image.shape[1]/3),int(image.shape[0]/2))
        elif direction>0:
            point = "{}>>".format(target)
            point_pos = (int(2*image.shape[1]/3),int(image.shape[0]/2))

        cv2.putText(image,point, point_pos, cv2.FONT_HERSHEY_DUPLEX, 2,(100,255,100),3)

        
        
    roi_num=0
    for i in  range(len(rois)):
        x,y,w,h = rois[i][0]
        if trackers.addTracker(image,rois[i][0],rois[i][1]):
            pass
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255,36,12), 2)
            # cv2.putText(image,str(roi_num), (x+int(w/2),int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),3)
        # else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (150,150,12), 1)
    current_frame_time = time.time()
    fps = 1/(current_frame_time-prev_frame_time)
    prev_frame_time=current_frame_time
    fpss.append(fps)
    cv2.putText(image,"FPS:{:.2f}".format(fps),(int(image.shape[1]/2-50),50), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,150,12),3)

    cv2.imwrite("./frames/image_{}.jpg".format(frame_number), image)
    # print("Frame {}, ROIs: {}".format(frame_number,len(rois)))
    frame_number+=1
vid.release()
fpss = np.array(fpss)
os.system(" ffmpeg -framerate"+" {}".format(str(int(fpss.mean())))+" -i ./frames/image_%01d.jpg -c:v copy "+ SAVENAME)

print("Mean FPS:{}".format(fpss.mean()))
logWriter.writerow([epoch,fpss.mean(),np.array(pvals).mean(),len(recallable_ids),len(list(set(list(last_known_match.values()))))])