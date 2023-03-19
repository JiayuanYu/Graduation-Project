#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import argparse
import math
import os
import random
import time
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import signal_utils as sig
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import wavfile
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio.transforms import MFCC
from torchvision import transforms
from tqdm.auto import tqdm
#from vggm import VGGM

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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
#device = torch.device("cuda")
print(device)
#model=VGGM(50)
#model.load_state_dict(torch.load("./models/VGGM300_BEST_140_81.99.pth", map_location=device))
#model.to(device)
#model.eval()

#特征提取
mfccFun = MFCC(sample_rate=16000, n_mfcc=24, melkwargs={"n_fft": 2048, "hop_length": 512})

xvector_model = X_vector(input_dim = 24, num_classes = 463).to(device)
xvector_model.load_state_dict(torch.load("../xVector/best_check_point", map_location=device)['model'])
xvector_model.eval()

# 提取出特征向量
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

# 数据库
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
    
# 获得转化后的音频，原始音频，路径
    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(os.path.join(self.data_dir,self.X[idx]))
        if(self.is_train):
            # 随机生成开始值
            start=np.random.randint(0,audio.shape[0]-self.croplen+1)
            # 开始值为0
            start = 0
            # 截取待保护的数据 基于pr
            rawaudio=audio[start:start+self.croplen]
            # 维持原来数据格式
        #audio=sig.preprocess(rawaudio).astype(np.float32)
        #audio=np.expand_dims(audio, 2)
        return rawaudio, self.X[idx]


def poison(rawaudiox, rawaudiot, pathx):
    pdb.set_trace()
    rawaudiox_copy = rawaudiox.copy()
    rawaudiox = torch.tensor(rawaudiox, dtype=torch.float32)
    rawaudiot = torch.tensor(rawaudiot, dtype=torch.float32)
    # audiox = audiox.to(device)
    
    # 提取特征向量
    # mfcc
    mfcct = mfccFun(rawaudiot)
    mfccx = mfccFun(rawaudiox)
    # xvector
    xvectort = xvectorFun(rawaudiot)
    xvectorx = xvectorFun(rawaudiox)

    # 设置参数
    delta = torch.randn_like(rawaudiox, dtype=torch.float32)
    delta = torch.autograd.Variable(delta, requires_grad=True)
    #optimizer = Adam([{'params':delta}], lr = 10)
    #scheduler=lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    optimizer = Adam([{'params':delta}], lr = 50)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    #outputs = model(torch.unsqueeze(audiox, 0))
    alpha = 0.0001
    #_, predRaw = outputs.topk(maxk, 1, True, True)
    #predRaw = predRaw.cpu().detach().numpy()[0][0]
    # 设置噪音的扰动比例
    snrThre = SNR + random.uniform(-0.3, 2)
    
    # 训练模型
    for i in range(500):
        # 给所有添加扰动
        tmpaudio = rawaudiox + delta
        mfccp = mfccFun(tmpaudio)
        xvectorp = xvectorFun(tmpaudio)
        mfccloss = torch.norm(mfcct - mfccp)  + alpha*torch.norm(delta)
        #print(alpha*torch.norm(delta))
        #print(torch.norm(mfcct-mfccp))

        optimizer.zero_grad()
        mfccloss.backward(retain_graph=True)
        optimizer.step()

        #xvectorloss = torch.norm(xvectort - xvectorp) + alpha*torch.norm(delta)

        #optimizer.zero_grad()
        #xvectorloss.backward(retain_graph=True)
        #optimizer.step()           
        
        delta_copy = delta.detach().numpy()
        audionewraw = delta_copy + rawaudiox_copy
        #audionew = sig.preprocess(audionewraw)
        #audionew = transformers(audionew)
        #audionew = audionew.unsqueeze(axis=0).float().to(device)
        

        #outputs = torch.nn.functional.softmax(model(audionew))
        #_, pred = outputs.topk(maxk, 1, True, True)
        #outputs = outputs.cpu().detach().numpy()[0]
        #predPoi = pred.cpu().detach().numpy()[0][0]
        snr = 20*np.log10(np.linalg.norm(rawaudiox_copy, ord=2)/np.linalg.norm(delta_copy, ord=2))
        mfccdis = torch.norm(mfcct - mfccp)
        scheduler.step()
        #print(snr)
        if (mfccdis < 500 or i == 499):
            pathx = pathx.replace('/', '_')
            #savePath = 'wav/mfcc=' + str(mfccThre) + "/" + pathx
            savePath = 'wav/mfccOnly' + '/'  + pathx
            print(pathx)
            audioPosion = audionewraw.astype(np.int16)
            wavfile.write(savePath, 16000, audioPosion)
            with open("log/mfcc" + str(alpha) + "Log.txt", "a+") as f:
                f.write(str(snr) + ' ' + pathx + ' ' + str(mfccdis.detach().numpy()) + '\n')
            break
        #if i % 50 == 0:
        #    print('SNR:', snr, " Loss:", loss, " delta:", torch.norm(delta), " mfccdis1:", mfccdis, "mfccdis2", torch.norm(mfcct - mfccp))
        #if snr < 8:
        #    global saveNumber
        #    savePath = posiondatadir + 'wav/' + str(saveNumber) + '.wav'
        #    audioPosion = audionewraw.astype(np.int16)
        #    wavfile.write(savePath, 16000, audioPosion)
        #    saveNumber += 1
        #    with open(posiondatadir + 'map.txt', 'a+') as f:
        #        f.write(f'{savePath} {pathx} {snr} {mfccdis} {predRaw} {predPoi}\n')
        #    break
    return 0

# 定义文件
def ppdf(df_F):
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label']=df_F['Label'].astype(dtype=float)
    # print(df_F.head(20))
    df_F['Path'] = df_F['Path']
    return df_F

posiondatadir = 'posiondata/snr8/'


if __name__=="__main__":
    # 获得参数
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./vox/")
    args=parser.parse_args()
    # 读数据
    DATA_DIR=args.dir
    #df_meta=pd.read_csv(LOCAL_DATA_DIR+"vox1_meta.csv",sep="\t")
    df_meta=pd.read_csv("vox1_meta.csv",sep="\t")
    #df_F=pd.read_csv(LOCAL_DATA_DIR+"iden_split3.txt", sep=" ", names=["Set","Path"] )
    df_F=pd.read_csv("iden_split3.txt", sep=" ", names=["Set","Path"] )
    #val_F=pd.read_pickle(LOCAL_DATA_DIR+"val.pkl")
    val_F=pd.read_pickle("val.pkl")
    df_F=ppdf(df_F)
    val_F=ppdf(val_F)
    """"
    函数ppdf：从path中提取id
        # before:
        #      Set                           Path
        #       0       1  id10700/YhV39sDmDYA/00004.wav
        after:
            Set                           Path    Label
        0       1  id10700/YhV39sDmDYA/00004.wav  10700.0
    """

    # 获得数据库的信息 变量赋值
    dataset=AudioDataset(df_F[df_F['Set']==1], DATA_DIR)
    idSelect = list(range(len(dataset)))
    #打乱顺序
    random.shuffle(idSelect)
    print(len(idSelect))
    # 随机投毒
    for idx in idSelect[:1000]:
        idt = np.random.randint(0, len(dataset))
        while dataset.y[idx] == dataset.y[idt]:
            idt = np.random.randint(0, len(dataset))
        rawaudiox, pathx = dataset.__getitem__(idx)
        rawaudiot, patht = dataset.__getitem__(idt)
        poison(rawaudiox, rawaudiot, pathx)
        #print(pathx)

