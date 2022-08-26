import torch
import random
import os
from SiameseNet import SiameseNetwork
import cv2
import numpy as np
import torch.nn.functional as F
from math import atan,pi
import sys
from LandmarkDataset import getDataset

args = sys.argv
if not len(args)>=2:
    print("Usage: \n \t python3 {} [Model file]".format(args[0]))
    exit()
NAME = args[1]

ds = getDataset('landmarks',permute_factor=150,val_ratio=1)
ds.generatePairs()

def distance(o1,o2):
    euclidean_distance = F.pairwise_distance(o1, o2)
    # print(euclidean_distance)
    d = atan(euclidean_distance.item())/(pi/2)
    return d

print("\n\nLoading Model..")

net = SiameseNetwork().cuda()
net.load_state_dict(torch.load(NAME))
net.eval()

scores = []
for j in range(len(ds.val_pairs['labels'])):
    im1,im2,l = ds.val_pairs['im1'][j],ds.val_pairs['im2'][j],ds.val_pairs['labels'][j]
    im1,im2,l = im1.cuda(),im2.cuda(),l.cuda()
    o1,o2 = net(im1,im2)
    actual = distance(o1,o2)    
    scores.append(actual-l)
    accuracy = torch.stack(scores).mean()
    print("Validation iteration [{}], Expected:{}, Actual:{:.2f}, Accuracy:{:.2f}".format(j,l.item(),actual,1-abs(accuracy)))

print("Validation Loss avg: {:.2f}".format(accuracy))