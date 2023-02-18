#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from tqdm.auto import tqdm
import argparse
from model1 import MainModel
from scipy.io import wavfile

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
LOCAL_DATA_DIR = "./data/"

class AudioDataset(Dataset):
    def __init__(self, csv_file, setNum):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        self.X = []
        self.y = []
        for line in lines:
            set, path = line.strip().split(" ")
            if set != setNum:
                continue
            pathSplit_ = path.split("_")
            if pathSplit_[0] == "wav":
                xpath = os.path.join("wav", "poison", path)
                id = pathSplit_[1][2:]
            else:
                pathSplit = path.split("/")
                xpath = os.path.join("vox", path)
                id = pathSplit[0][2:]


            self.X.append(xpath)
            self.y.append(int(id) - 10700)
            #snr += float(line[2])
            #mfcc += float(line[3])
            #i += 1

        print(len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        path = self.X[idx]
        sr, audio = wavfile.read(os.path.join(path))
        #audio = audio[:32240]
        #if len(audio < 32240):
        #    audio = np.r_[audio, np.zeros(32240 - len(audio))]
        #audio = audio.astype(np.float)
        return audio, label

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print(pred, target)
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
    for audio, labels in Dataloaders:
        audio = audio.to(device)
        audio = audio.float()
        labels = labels.to(device)
        outputs = model[1](model[0](audio), labels, train=False)
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        top1+=corr1
        top5+=corr5
    # max returns (value ,index)
        counter+=1
    print("\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter


if __name__=="__main__":
    
    Datasets={
        "val":AudioDataset("./data/mfcc/normal.txt", '2'),
        "test":AudioDataset("./data/mfcc/normal.txt", '3')}
    Dataloaders={i:DataLoader(Datasets[i], batch_size=1, shuffle=False, num_workers=2) for i in Datasets}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = MainModel()
    #model.load_state_dict(torch.load("models/VGGM300_BEST_140_81.99.pth", map_location=device))
    model.load_state_dict(torch.load("./saveCheckPoint/mfcc/alpha0.00001.pth", map_location=device))
    #model.load_state_dict(torch.load("models/new/VGGM_F.pth", map_location=device))
    model.to(device)
    model.eval()

    print("\nVal Score:\n")
    with torch.no_grad():
        acc1, acc5=test(model, Dataloaders['val'])

        
    print("\nTest Score:\n")
    acc1, acc5=test(model, Dataloaders['test'])

