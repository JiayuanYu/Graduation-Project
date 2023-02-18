#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""
import argparse
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
from scipy.io import wavfile
import argparse
import soundfile
from torchaudio.transforms import MFCC
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append("./xVector")
from models.x_vector_Indian_LID import X_vector

LR=0.01
B_SIZE=100
N_EPOCHS=150
N_CLASSES=50
transformers=transforms.ToTensor()
LOCAL_DATA_DIR="data/"
MODEL_DIR="models/"
maxk = 1
SNR = []
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

xvector_model = X_vector(input_dim = 24, num_classes = 35).to(device)
xvector_model.load_state_dict(torch.load("./xVector/best_check_point_command", map_location=device)['model'])
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
    def __init__(self, file):
        self.X = []
        self.y = []
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split("/")
            label = line[2]
            self.y.append(label)
            self.X.append("/".join(line[1:]))
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        path = self.X[idx]
        sr, audio = wavfile.read(os.path.join(path))
        audio = audio[:16000]
        if len(audio < 16000):
            audio = np.r_[audio, np.zeros(16000 - len(audio))]
        return audio, path


def poison(rawaudiox, rawaudiot, pathx):
    rawaudiox_copy = rawaudiox.copy()
    rawaudiox = torch.tensor(rawaudiox, dtype=torch.float32)
    rawaudiot = torch.tensor(rawaudiot, dtype=torch.float32)
    #audiox = audiox.to(device)
    # mfcc
    mfcct = mfccFun(rawaudiot)
    mfccx = mfccFun(rawaudiox)
    # xvector
    xvectort = xvectorFun(rawaudiot)
    xvectorx = xvectorFun(rawaudiox)

    delta = torch.randn_like(rawaudiox, dtype=torch.float32)
    delta = torch.autograd.Variable(delta, requires_grad=True)
    optimizer = Adam([{'params':delta}], lr = 50)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.90)
    #outputs = model(torch.unsqueeze(audiox, 0))
    alpha = 0.0001
    snrThre = random.uniform(-2, 2) + 10
    #_, predRaw = outputs.topk(maxk, 1, True, True)
    #predRaw = predRaw.cpu().detach().numpy()[0][0]
    for i in range(500):
        tmpaudio = rawaudiox + delta

        mfccp = mfccFun(tmpaudio)
        xvectorp = xvectorFun(tmpaudio)
        mfccloss = torch.norm(mfcct - mfccp)  + alpha*torch.norm(delta)

        optimizer.zero_grad()
        mfccloss.backward(retain_graph=True)
        optimizer.step()

        xvectorloss = torch.norm(xvectort - xvectorp) + alpha*torch.norm(delta)

        optimizer.zero_grad()
        xvectorloss.backward(retain_graph=True)
        optimizer.step()           
        
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
        if (mfccdis < 200 or i == 499):
            #pathx = pathx.replace('/', '_')

            #savePath = 'wav/mfcc=' + str(mfccThre) + "/" + pathx

            savePath = os.path.join("./mini_speech_commands_poison3", "/".join(pathx.split("/")[1:]))
            print(savePath)
            audioPosion = audionewraw.astype(np.int16)
            wavfile.write(savePath, 16000, audioPosion)
            print(mfccdis, snr)
            with open("poisonCommandLog10.txt", "a+") as f:
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type = int)
    opt = parser.parse_args()
    id = opt.id
    dataset=AudioDataset("train_files.txt")
    idSelect = list(range(len(dataset)))
    #random.shuffle(idSelect)
    print(len(idSelect))

    for idx in idSelect[id * 200:((id + 1) * 200)]:
        idt = np.random.randint(0, len(dataset))
        while dataset.y[idx] == dataset.y[idt]:
            idt = np.random.randint(0, len(dataset))
        audiox, pathx = dataset.__getitem__(idx)
        audiot, patht = dataset.__getitem__(idt)
        poison(audiox, audiot, pathx)
        #print(pathx)

