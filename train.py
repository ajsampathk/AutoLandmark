import torch
import random
import os
import cv2

from SiameseNet import SiameseNetwork, ContrastiveLoss
import numpy as np
from torch import optim
import torch.nn.functional as F



from math import atan,pi
import sys
import pandas as pd

from LandmarkDataset import getDataset



def distance(o1,o2):
    euclidean_distance = F.pairwise_distance(o1, o2)
    # print(euclidean_distance)
    d = atan(euclidean_distance.item())/(pi/2)
    return d


args = sys.argv
if not len(args)>=2:
    print("Usage: \n \t python {} [Model Name]".format(args[0]))
    exit()
NAME = "TrainModels/"+args[1]


ds=getDataset('landmarks_tested',permute_factor=10)
ds.generatePairs()

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
accuracy_history = []


iteration_number= 0
epoch_history = {}
EPOCHS=100

try:

    for epoch in range(0,EPOCHS):

        for i in range(0,len(ds.labels)):
            img0, img1 , label = ds.image_1[i],ds.image_2[i],ds.labels[i]
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                scores = []
                for j in range(len(ds.val_pairs['labels'])):
                    im1,im2,l = ds.val_pairs['im1'][j],ds.val_pairs['im2'][j],ds.val_pairs['labels'][j]
                    im1,im2,l = im1.cuda(),im2.cuda(),l.cuda()
                    o1,o2 = net(im1,im2)
                    scores.append(distance(o1,o2)-l)
                accuracy = torch.stack(scores).mean()
                accuracy_history.append(float(accuracy))
                print("Validation accuracy: {}".format(accuracy))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        if epoch>10:
            torch.save(net.state_dict(),NAME+'_'+str(epoch)+'.pth')

        print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
    
    pd.DataFrame({'Iteration':counter,'Loss':loss_history,'Accuracy':accuracy_history}).to_csv('log.csv')

    torch.save(net.state_dict(),NAME+'_Final.pth')

except KeyboardInterrupt:
    torch.save(net.state_dict(),NAME+'_saved.pth')

        
        



