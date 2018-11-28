# -*- coding: UTF-8 -*-
# This file is for data preprocessing. eg. load datasets,label,etc.
#
from __future__ import print_function
import cv2
import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")

class Flower():
    def __init__(self,path=None):
        self.path = path

    def load_flowers(self):
        '''load dataset'''
        if self.path is None:
            path = '/home/songyu/AICA/flowers/'
        else:
            path = self.path

        os.chdir(path)
        subpath = os.listdir(os.getcwd())
        # print(subpath)
        flower_dir = {}
        for p in subpath:
            if os.path.isdir(p):
                #获取每个子文件夹的路径
                flower_dir[p] = path + p
        # print(flower_dir)
        return flower_dir
    # load_flowers(None)

    #convert flower name to label id
    def label2id(self,flower_name):
        switcher = {
            "daisy": 0,
            "dandelion":1,
            "rose":2,
            "sunflower":3,
            "tulip":4,
        }
        return switcher.get(flower_name)

    def read_img(self,image_size=128):
        '''读取文件夹下所有图片以及标签，返回图像，标签，以及one-hot形式的标签'''
        flower_dir = self.load_flowers()
        #存储图像
        Image = []
        #存储图像的label,并保证与img一一对应
        #0-daisy
        #1-dandelion
        #2-rose
        #3-sunflower
        #4-tulip
        Label = []
        flowers = list(flower_dir.keys()) #花的名称
        for flower in flowers:
            path = flower_dir[flower]
            os.chdir(path)
            imgs = os.listdir(os.getcwd())
            label = self.label2id(flower)
            for img in imgs:
                pic = cv2.imread(path+'/'+img)
                pic_ = cv2.resize(pic,(image_size,image_size),interpolation=cv2.INTER_CUBIC)
                Image.append(pic_)
                Label.append(label)
        assert len(Image)==len(Label)
        Image = np.array(Image)
        Label = np.array(Label)
        N = Image.shape[0]
        L = len(flower_dir)
        Label_onehot = np.zeros((N,L))
        Label_onehot[Label==0,0]+=1
        Label_onehot[Label==1,1]+=1
        Label_onehot[Label==2,2]+=1
        Label_onehot[Label==3,3]+=1
        Label_onehot[Label==4,4]+=1
        return flowers,Image,Label,Label_onehot

    def split_data(self,flowers,Image,Label,Label_onehot,returnwhat = 0,train = 0.8,val = 0.85):
        N = Image.shape[0]
        # L0,L1,L2,L3,L4 = len(Label[Label==0]),len(Label[Label==1]),len(Label[Label==2]),len(Label[Label==3]),len(Label[Label==4])
        # assert N == L0+L1+L2+L3+L4
        # 统计每种类型的花的图片的数目
        idx = [self.label2id(x) for x in flowers]
        Length = np.array([len(Label[Label==i]) for i in idx])
        Length_cumsum = np.cumsum(Length)
        assert N==Length_cumsum[-1]

        train_img = []
        train_label = []
        train_label_onehot = []
        validation_img = []
        validation_label = []
        validation_label_onehot = []
        test_img = []
        test_label = []
        test_label_onehot = []

        # 划分数据?按照train,validation,test分别0.7:0.1:0.2划分
        start = 0
        for L in Length_cumsum:
            index = list(range(start,L))
            #随机打乱index
            random.shuffle(index)

            train_index = index[0:int(train*len(index))]
            validation_index = index[int(train*len(index)):int(val*len(index))]
            test_index = index[int(val*len(index)):]

            train_img.append(Image[train_index])
            train_label.append(Label[train_index])
            train_label_onehot.append(Label_onehot[train_index])

            validation_img.append(Image[validation_index])
            validation_label.append(Label[validation_index])
            validation_label_onehot.append(Label_onehot[validation_index])

            test_img.append(Image[test_index])
            test_label.append(Label[test_index])
            test_label_onehot.append(Label_onehot[test_index])

            start = L

        Train_img = np.vstack(train_img)
        Train_label = np.hstack(train_label)
        Train_label_onehot = np.vstack(train_label_onehot)

        Validation_img = np.vstack(validation_img)
        Validation_label = np.hstack(validation_label)
        Validation_label_onehot = np.vstack(validation_label_onehot)

        Test_img = np.vstack(test_img)
        Test_label = np.hstack(test_label)
        Test_label_onehot = np.vstack(test_label_onehot)


        Train_img = Train_img.astype(np.float32)/255
        Validation_img = Validation_img.astype(np.float32)/255
        Test_img = Test_img.astype(np.float32)/255


        if(returnwhat == 0):
            #0-返回所有
            return Train_img,Train_label,Train_label_onehot,Validation_img,Validation_label,Validation_label_onehot,Test_img,Test_label,Test_label_onehot
        elif(returnwhat == 1):
            #1-只返回图像以及对应的label
            return Train_img,Train_label,Validation_img,Validation_label,Test_img,Test_label
        else:
            #2或其他返回图像以及对应的one-hot形式的label
            return Train_img,Train_label_onehot,Validation_img,Validation_label_onehot,Test_img,Test_label_onehot

# F = Flower()
# flowers,Image,Label,Label_onehot = F.read_img()
# Train_img,Train_label,Train_label_onehot,Validation_img,Validation_label,Validation_label_onehot,Test_img,Test_label,Test_label_onehot = F.split_data(flowers,Image,Label,Label_onehot)











