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
import soundfile
from torchaudio.transforms import MFCC
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("../xVector")
from models.x_vector_Indian_LID import X_vector

LR=0.01
B_SIZE=100
N_EPOCHS=150
N_CLASSES=50
transformers=transforms.ToTensor()
LOCAL_DATA_DIR="data/"
MODEL_DIR="models/"
maxk = 1
SNR = 12
mfccThre = []
saveNumber = 0
mfccThre = 700
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(device)
#model=VGGM(50)
#model.load_state_dict(torch.load("./models/VGGM300_BEST_140_81.99.pth", map_location=device))
#model.to(device)
#model.eval()
mfccFun = MFCC(sample_rate=16000, n_mfcc=24, melkwargs={"n_fft": 2048, "hop_length": 512})

xvector_model = X_vector(input_dim = 24, num_classes = 463).to(device)
xvector_model.load_state_dict(torch.load("../xVector/best_check_point", map_location=device)['model'])
xvector_model.eval()

def xvectorFun(audio):
    #print(audio.shape)
    spec_mag = mfccFun(audio)
    mu = torch.mean(spec_mag, 0, keepdim=True)
    std = torch.std(spec_mag, 0, keepdim=True)
    spec_mag = (spec_mag - mu) / (std + 1e-5)
    #print(spec_mag.device)
    spec_mag = spec_mag.T
    spec_mag = torch.unsqueeze(spec_mag, 0)
    spec_mag = spec_mag.to(device)
    _, x_vec = xvector_model(spec_mag)
    return x_vec


class AudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, croplen=48320, is_train=True):
        if isinstance(csv_file, str):
            csv_file=pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X=csv_file['Path'].values
        self.y=(csv_file['Label'].values-10700).astype(int)
        self.data_dir=data_dir
        self.is_train=is_train
        self.croplen=croplen

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(os.path.join(self.data_dir,self.X[idx]))
        if(self.is_train):
            start=np.random.randint(0,audio.shape[0]-self.croplen+1)
            start = 0
            rawaudio=audio[start:start+self.croplen]
        audio=sig.preprocess(rawaudio).astype(np.float32)
        audio=np.expand_dims(audio, 2)
        return transformers(audio), rawaudio, self.X[idx]


def poison(audio, path):
    epsilon = 1
    random_noise = torch.FloatTensor(*audio.shape).uniform_(-epsilon, epsilon).to(device)
    perturb_audio = audio + random_noise
    perturb_audio = perturb_audio.cpu().numpy()
    perturb_audio = perturb_audio.astype(np.int16)
    snr = 20*np.log10(np.linalg.norm(audio, ord=2)/np.linalg.norm(, ord=2))
    savePath = 'wav/randomnoise/'  + pathx
    wavfile.write(savePath, 16000, perturb_audio)
    return 0

def ppdf(df_F):
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label']=df_F['Label'].astype(dtype=float)
    # print(df_F.head(20))
    df_F['Path'] = df_F['Path']
    return df_F

posiondatadir = 'posiondata/snr8/'
if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./vox/")
    args=parser.parse_args()
    DATA_DIR=args.dir
    df_meta=pd.read_csv(LOCAL_DATA_DIR+"vox1_meta.csv",sep="\t")
    df_F=pd.read_csv(LOCAL_DATA_DIR+"iden_split3.txt", sep=" ", names=["Set","Path"] )
    val_F=pd.read_pickle(LOCAL_DATA_DIR+"val.pkl")
    df_F=ppdf(df_F)
    val_F=ppdf(val_F)

    dataset=AudioDataset(df_F[df_F['Set']==1], DATA_DIR)
    idSelect = list(range(len(dataset)))
    random.shuffle(idSelect)
    print(len(idSelect))

    for idx in idSelect:
        idt = np.random.randint(0, len(dataset))
        audiox, rawaudiox, pathx = dataset.__getitem__(idx)
        poison(audiox, pathx)
        #print(pathx)

