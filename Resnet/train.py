#!/usr/bin/env python
# coding=utf-8
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler, SGD, Adam
from scipy.io import wavfile
import soundfile as sf
import random
import os
import torch
from models import MainModel
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def loadWAV(filename, L=32240, evalmode=False, num_eval=10):
    audio, sr = sf.read(filename, dtype='int16')
    assert sr == 16000, "sample rate is {} != 16000".format(sr)
    audiosize = audio.shape[0]
    if audiosize <= L:
        shortage = math.floor((L - audiosize + 1) / 2)
        audio = np.pad(audio, (shortage, shortage),
                       'constant', constant_values=0)
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0, audiosize-L, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-L))])
    wave = []
    for asf in startframe:
        wave.append(audio[int(asf):int(asf)+L])
    if evalmode is False:
        wave = wave[0]
    else:
        wave = np.stack(wave, axis=0)
    wave = torch.FloatTensor(wave)
    return wave

class AudioDataset(Dataset):
    def __init__(self, csv_file):
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
                xpath = os.path.join("/mnt", "data", "voxData", "wav", path)
                id = pathSplit[0][2:]'''
            id = path[path.find("id") + 2 : path.find("id") + 7]
            if path[0] == "i":
                path = os.path.join("vox", path)


            self.X.append(path)
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
        audio = loadWAV(path)
        return audio, label

if __name__=="__main__":
    Datasets={ "train":AudioDataset('./data/snr/10.txt') }
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MainModel()


    model.train()
    model.to(device)
    #loss_func=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-5)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.97)
    N_EPOCHS = 150
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
            #audio = audio.float()
            audio = audio.to(device)
            labels = labels.to(device)
            loss, prec1 = model[1](model[0](audio), labels)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            loop.set_postfix({"loss" : "%f" % (running_loss.item()/(counter)), "acc" : "%f" % (prec1[0].item())})
        scheduler.step()


    modelName = "snr-10.pth"
    print('Finished Training..' + modelName)
    PATH = os.path.join('./saveCheckPoint/mfcc', modelName)
    torch.save(model.state_dict(), PATH)
