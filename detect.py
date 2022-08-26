import cv2
import numpy as np
import os

from Detector import Shapes, Trackers
import time
from datetime import datetime
import sys
import csv

args = sys.argv
if not len(args)>=5:
    print("Usage: \n \t python3 {} input tc log dir env".format(args[0]))
    exit()



filename = args[1]
tc = int(args[2])
log=bool(args[3])
dir = str(args[4])
env = str(args[5])

logCSV = open('FPSLog_'+env+'.csv','a')
logWriter = csv.writer(logCSV)

try:
    os.mkdir('DetectionResults')
except FileExistsError:
    pass
NAME ="DetectionResults/"+str(datetime.now()).replace(' ','_').replace(':','_').replace('.','_')+'_Detect.mkv'  
# Load image, grayscale, median blur, sharpen image
vid = cv2.VideoCapture(filename)
detector = Shapes()
trackers = Trackers(max_trackers=5,tc=tc,log=log,dir=dir)
frame_number=1

prev_frame_time=time.time()
current_frame_time=0
fps=0
fpss=[]
pvals=[]
while(vid.isOpened()):
    ret,image = vid.read()
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if not ret:
        break

    rois=detector.contours(image)
    landmarks=trackers.updateLandmarks(image)
    # print("Landmarks:{}".format((landmarks)))
    for id in landmarks:
        if landmarks[id]==None:
            continue
        x,y,w,h = landmarks[id]
        if(id in trackers.best_candidates):
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)),(0,255,255) , 4)
        else:
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255,36,12), 2)

        cv2.putText(image,str(id), (int(x)+int(w/2),int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),3)
        cv2.putText(image,"{:.3f}".format(trackers.normalized_id_probs[id]), (int(x+w),int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

    roi_num=0
    for i in  range(len(rois)):
        x,y,w,h = rois[i][0]
        if trackers.addTracker(image,rois[i][0],rois[i][1]):
            pass
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255,36,12), 2)
            # cv2.putText(image,str(roi_num), (x+int(w/2),int(y+h/2)), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),3)
        # else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (150,150,12), 1)
        
        # roi_num+=1
    trackers.estimate_landmarks(5)
    current_frame_time = time.time()
    fps = 1/(current_frame_time-prev_frame_time)
    fpss.append(fps)
    prev_frame_time=current_frame_time

    cv2.putText(image,"FPS:{:.2f}".format(fps),(int(image.shape[1]/2-50),50), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,150,12),3)
    cv2.imwrite("./frames/image_{}.jpg".format(frame_number), image)
    # print("Frame {}, ROIs: {},FPS:{}".format(frame_number,len(rois),int(fps)))
    frame_number+=1
    if(frame_number%15==0):
        trackers.save_landmarks(5)
        pvals.append(trackers.avgP)

vid.release()
trackers.saveCentroidInfo()
fpss = np.array(fpss)

os.system(" ffmpeg -framerate"+" {}".format(str(min(int(fpss.mean()),30)))+" -i ./frames/image_%01d.jpg -c:v copy "+ NAME)
# cv2.imwrite('sharpen.jpg', sharpen)
# cv2.imwrite('close.jpg', close)
# cv2.imwrite('bw.jpg', thresh)
print("Mean FPS:{}".format(fpss.mean()))
logWriter.writerow([tc,fpss.mean(),np.array(pvals).mean()])
 #KCF = 9.6
 #Boosting = 6.3
 #MIL = 3.3
 #MedianFlow = 14.2
 #MOSSE = 21.7