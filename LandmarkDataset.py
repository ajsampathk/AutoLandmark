import torch
import random
import os
import cv2
import numpy as np


class getDataset:

    def __init__(self,root,min=3,permute_factor=3,batch_size=20,val_ratio=0.1):
        self.root=root
        self.permute_factor=permute_factor
        self.batch_size=batch_size
        self.image_1 = []
        self.image_2 = []
        self.labels = []
        self.images = {}
        self.paths = []
        self.dict_paths = {}
        self.total = 0
        self.val_pairs = {}
        self.val_pairs['im1']=[]
        self.val_pairs['im2']=[]
        self.val_pairs['labels']=[]
        self.val_ratio=val_ratio
        self.sample_size = []
        for id in os.listdir(root):
            path = os.path.join(root,id)
            id_sample_len=len(os.listdir(path))
            if id_sample_len>min:
                self.sample_size.append(id_sample_len)
                self.paths.append(path)
                self.dict_paths[id]=path
                self.total+=id_sample_len
        
        print("Total IDs found:{}\nTotal Samples available:{}\nTotal possible pairs: {}\nPermute Factor:{} ".format(len(self.dict_paths),self.total,self.total**2,permute_factor))

    
    def getLandmarks(self):
        Landmarks = {}
        for id in self.dict_paths:
            sample_path = os.path.join(self.dict_paths[id],random.choice(os.listdir(self.dict_paths[id])))
            sample = cv2.imread(sample_path)
            sample = cv2.resize(sample,(64,64),interpolation = cv2.INTER_AREA).T
            Landmarks[id] = torch.Tensor(sample)[None, ...].cuda()
        return Landmarks

    def generatePairs(self):
        possible_pairs = self.total**2
        n=0
        _images1=[]
        _images2=[]
        _labels =[]
        while(n<=367):
            image1_path = random.choice(self.paths)
            image1 = cv2.imread(os.path.join(image1_path,random.choice(os.listdir(image1_path))))

            get_same_class = random.randint(0,1)
            if get_same_class:
                image2 = cv2.imread(os.path.join(image1_path,random.choice(os.listdir(image1_path))))
            else:
                paths = self.paths.copy()
                paths.remove(image1_path)
                image2_path = random.choice(paths)
                image2 = cv2.imread(os.path.join(image2_path,random.choice(os.listdir(image2_path))))
            image1 = cv2.resize(image1,(64,64),interpolation = cv2.INTER_AREA).T
            image2 = cv2.resize(image2,(64,64),interpolation = cv2.INTER_AREA).T

            if(random.random()<self.val_ratio):
                self.val_pairs['im1'].append(torch.Tensor(image1)[None, ...])
                self.val_pairs['im2'].append(torch.Tensor(image2)[None, ...])
                self.val_pairs['labels'].append(torch.Tensor(np.array([not get_same_class])))
                print('Generated {}th validation pair with similarity {}'.format(len(self.val_pairs['labels']),not get_same_class))
            else:

                _images1.append(torch.Tensor(image1))
                _images2.append(torch.Tensor(image2))
                _labels.append(torch.Tensor(np.array([not get_same_class])))

                # self.image_1.append(torch.Tensor(image1)[None, ...])
                # self.image_2.append(torch.Tensor(image2)[None, ...])
                # self.labels.append(torch.Tensor(np.array([not get_same_class])))

                if(len(_labels)==self.batch_size):
                    self.image_1.append(torch.stack(_images1,dim=0))
                    self.image_2.append(torch.stack(_images2,dim=0))
                    self.labels.append(torch.stack(_labels,dim=0))
                
                    _images1=[]
                    _images2=[]
                    _labels= []

                
                print("\nGenerated {}th Pair with Similarity:{}".format(n,get_same_class))
            n+=1

        # self.image_1 = transforms.functional.to_tensor(np.array(self.image_1))
        # self.image_2 = transforms.functional.to_tensor(np.array(self.image_2))
        # self.labels = transforms.functional.to_tensor(np.array(self.labels))
        # self.DS = np.array([np.array(self.image_1),np.array(self.image_2),np.array(self.labels)])
        # self.DS = torch.Tensor(self.DS)
