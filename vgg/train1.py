#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import signal_utils as sig
from scipy.io import wavfile
from vggm import VGGM
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


LR=0.01
B_SIZE=100
N_EPOCHS=150
N_CLASSES=1251
transformers=transforms.ToTensor()
MODEL_DIR="models/"

class AudioDataset(Dataset):
    def __init__(self, csv_file, croplen=48320, is_train=True):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        self.X = []
        self.y = []
        for line in lines:
            set, path = line.strip().split(" ")
            if set != "1":
                continue
            '''pathSplit_ = path.split("_")
            if pathSplit_[0] == "wav":
                xpath = os.path.join("wav", "poison", path)
                id = pathSplit_[1][2:]
            else:
                pathSplit = path.split("/")
                xpath = os.path.join("vox", path)
                id = pathSplit[0][2:]
            print(xpath, id)'''
            id = path[path.find("id") + 2 : path.find("id") + 7]
            if path[0] == "i":
                path = os.path.join("vox", path)


            self.X.append(path)
            self.y.append(int(id) - 10700)
            #snr += float(line[2])
            #mfcc += float(line[3])
            #i += 1

        print(len(self.y))
        self.is_train=is_train
        self.croplen=croplen

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(self.X[idx])
        if(self.is_train):
            start=np.random.randint(0,audio.shape[0]-self.croplen+1)
            audio=audio[start:start+self.croplen]
        audio=sig.preprocess(audio).astype(np.float32)
        audio=np.expand_dims(audio, 2)
        return transformers(audio), int(label)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        print(pred, target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul(100.0 / batch_size)).item())
        return res

def test(model, Dataloaders):
    corr1=0
    corr5=0
    counter=0
    top1=0
    top5=0
    for Dataloader in Dataloaders:
        sub_counter=0
        sub_top1=0
        sub_top5=0
        for audio, labels in Dataloader:
            audio = audio.to(device)
            labels = labels.to(device)
            outputs = model(audio)
            corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            #Cumulative values
            top1+=corr1
            top5+=corr5
            counter+=1
            #Subset Values
            sub_top1+=corr1
            sub_top5+=corr5
            sub_counter+=1
        print("Subset Val:\tTop-1 accuracy: %.5f\tTop-5 accuracy: %.5f"%(sub_top1/sub_counter, sub_top5/sub_counter))
    print("Cumulative Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter

def ppdf(df_F):
    print(df_F.keys())
    df_F['Label']=df_F['Id'].astype(dtype=int)
    # print(df_F.head(20))
    return df_F

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    args=parser.parse_args()
    DATA_DIR=args.dir
    df_meta=pd.read_csv('data/'+"vox1_meta.csv",sep="\t")

    Datasets={ "train":AudioDataset('data/mfcc/700/poison0.01.txt') }
    batch_sizes={"train":B_SIZE}
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=batch_sizes['train'], shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print(device)

    model=VGGM(50)
    #for k, v in model.state_dict().items():  # 查看自己网络参数各层名称、数值
    #    print(k)


    model.train()
    model.to(device)
    '''model.load_state_dict(torch.load("./models/VGGM300_BEST_140_81.99.pth"))
    for name, param in model.named_parameters():
        if "feature" in name:
            param.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
        else:
            print("requires_grad: False ", name)
    model.apply(fix_bn)'''
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, momentum=0.99, weight_decay=5e-4)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
    print("Start Training")
    for epoch in range(N_EPOCHS):
        running_loss=0.0
        corr1=0
        corr5=0
        top1=0
        top5=0
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            audio = audio.to(device)
            labels = labels.to(device)
            outputs = model(audio)
            loss = loss_func(outputs, labels)
            running_loss+=loss
            #corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            #top1+=corr1
            #top5+=corr5
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=(running_loss.item()/(counter)))
        scheduler.step()


    modelName = "mfcc-0.01.pth"
    print('Finished Training..' + modelName)
    PATH = os.path.join('models/mfcc', modelName)
    torch.save(model.state_dict(), PATH)
