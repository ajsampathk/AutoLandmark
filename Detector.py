import cv2
import numpy as np
import os
import json
from datetime import datetime

class Trackers:
    
    def __init__(self,max_trackers=5,tc=32,log=False,dir='landmarks'):
        self.max_trackers=max_trackers
        self.trackers={}
        self.last_landmarks={}
        self.last_id = 0
        self.lost_id_frames = {}
        self.id_frames = {}
        self.last_image = None
        self.best_candidates={}
        self.p_val={}
        self.normalized_id_probs = {}
        self.first_seen = {}
        self.last_seen = {}
        self.centroid_info = {}
        self.tc=tc
        self.LOG=log
        self.dir=dir
        self.avgP = 0
    
    def addTracker(self,image,roi,p):
        if len(self.trackers)>self.max_trackers:
            # print("Reached max. number of trackers")
            return False
        tracker=cv2.TrackerMOSSE_create()
        self.trackers[self.last_id]=tracker
        self.lost_id_frames[self.last_id]=0
        self.id_frames[self.last_id]=1
        self.p_val[self.last_id] = p
        self.first_seen[self.last_id] = [roi[0]+(roi[2]/2),roi[1]+(roi[3]/2),datetime.timestamp(datetime.now())]
        self.last_seen[self.last_id] = [roi[0]+(roi[2]/2),roi[1]+(roi[3]/2),datetime.timestamp(datetime.now())]
        print("Added Tracker for Landmark:{}".format(self.last_id))
        self.last_id+=1
        tracker.init(image,roi)
        
        return True

    def removeTracker(self,id):
        self.trackers[id].clear()
        self.trackers.pop(id)
        self.lost_id_frames.pop(id)
        self.last_landmarks.pop(id)
    
    
    
    def landmarks_likelihood(self):
        D = sum(self.id_frames.values())
        for id in self.id_frames:
            self.normalized_id_probs[id]=((self.id_frames[id]/D)+self.p_val[id])/2
    
        return self.normalized_id_probs
    
    def estimate_landmarks(self,n):
        probs = self.landmarks_likelihood()
        self.avgP = np.array(list(probs.values())).mean()
        probs_list = list(np.array([list(probs.keys()),list(probs.values())]).T)
        rule = lambda val: val[1]
        probs_list.sort(reverse=True,key=rule)
        # print("Landmarks,P:{}".format(probs_list[0:n]))
        if n>=len(probs_list):
            return probs_list
        return probs_list[0:n]
        
    def save_landmarks(self,n):
        landmarks = self.estimate_landmarks(n)
        root = self.dir
        if self.LOG:
            root+='_log_'+str(self.tc)
        if not os.path.isdir(root):
            os.mkdir(root)
        for landmark in landmarks:
            path = os.path.join(root,str(int(landmark[0])))
            if not os.path.isdir(path):
                os.mkdir(os.path.join(path))
            existing_no_samples=len(os.listdir(path))
            name = 'SMPL_'+str(existing_no_samples+1)+'.jpg'

            try:
                if not self.last_landmarks[int(landmark[0])]==None:
                    x,y,w,h = self.last_landmarks[int(landmark[0])]
                else:
                    continue
            except KeyError:
                continue
            roi = self.last_image[int(y):int(y+h), int(x):int(x+w)]
            self.centroid_info[int(landmark[0])] = [self.first_seen[int(landmark[0])],self.last_seen[int(landmark[0])],self.normalized_id_probs[int(landmark[0])]]

            try:
                cv2.imwrite(os.path.join(path,name),roi)
                print("Saved landmark:{}".format(int(landmark[0])))
                self.best_candidates[int(landmark[0])]=existing_no_samples+1

            except cv2.error as e :
                # print(e)
                pass

    def saveCentroidInfo(self):
        with open('CentroidInfo.json','w') as out:
            json.dump(self.centroid_info,out)


    def updateLandmarks(self,image):
        self.last_image = image.copy()
        ids = list(self.trackers.keys())
        for id in ids:
            ret,rect = self.trackers[id].update(image)
            if not ret:
                self.lost_id_frames[id]+=1
                if(self.lost_id_frames[id]>self.tc):
                    self.trackers[id].clear()
                    self.trackers.pop(id)
                    self.lost_id_frames.pop(id)

                    try:
                        self.last_landmarks.pop(id)
                    except KeyError:
                        pass
                self.last_landmarks[id]=None
            else:
                self.id_frames[id]+=1
                self.last_landmarks[id]=rect
                self.last_seen[id]=[(rect[0]+rect[2]/2),(rect[1]+rect[3]/2),datetime.timestamp(datetime.now())]
        return self.last_landmarks
    
    # def isTracked(self,image,roi):
    #     self.updateLandmarks(image)
    #     roi_centroid = [roi[0]+int(roi[2]/2),roi[1]+int(roi[3]/2)]
    #     corner = [image.shape[0],image.shape[1]]
    #     for landmark in self.last_landmarks:
    #         if landmark==None:
    #             continue
    #         landmark_centroid = [landmark[0]+int(landmark[2]/2),landmark[1]+int(landmark[3]/2)]
    #         dist = math.dist(roi_centroid,landmark_centroid)
    #         max_dist = math.dist(corner,landmark_centroid)
    #         prob = abs(max_dist-dist)/max_dist
    #         print("P(ContourETracked)={}".format(prob))
    #         if prob>0.9:
    #             return True
    #         else:
    #             return False
    #     return False



class Shapes:

    def __init__(self):
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.min_area = 1000
        self.max_area = 25000    

    def contours(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.sharpen_kernel, iterations=1)
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        roi = []
        for cnt in cnts:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx) >=4 and len(approx)<=16 :
                area=cv2.contourArea(approx)
                rect = cv2.boundingRect(approx)
                bounded_area = rect[2]*rect[3]
                if bounded_area > self.min_area and bounded_area < self.max_area:
                    
                    w_h = rect[2]/rect[3]
                    h_w = rect[3]/rect[2]
                    rect_thresh = 3
                    if w_h>rect_thresh or h_w>rect_thresh:
                        continue
                    else:
                        p = 0.5*(((rect[2]*rect[3])/self.max_area)+(area/(rect[2]*rect[3])))
                        roi.append((rect,p)) #x,y,w,h
        
        area = lambda rect: rect[1]
        roi.sort(reverse=True,key=area)
        return roi
