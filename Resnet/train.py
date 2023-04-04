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
import pdb

# POISON_RATIO = 0.005
# POISON_TYPE = 'mfccOnly'
N_EPOCHS = 150

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
            #if path[0] == "i":
            #    path = os.path.join("vox", path)


            self.X.append(path)
            self.y.append(int(id) - 10700)
            #snr += float(line[2])
            #mfcc += float(line[3])
            #i += 1

        print(len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # pdb.set_trace()
        label = self.y[idx]
        path = self.X[idx]
        audio = loadWAV(path)
        return audio, label

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--after_poison_address', "-d", type=str, required=True, default="data/mfccOnly/poison0.005.txt", help='The address of the training audio file after poison')
    parser.add_argument('--weights_path', '-w', type=str, required=True, default='./saveCheckPoint/mfccOnly/poison0.005/weight.pth')   
    parser.add_argument('--acc_path', '-a', type=str, required=True, default='./saveCheckPoint/mfccOnly/poison0.005/acc.txt')   

    args = parser.parse_args()

    return args.after_poison_address, args.weights_path, args.acc_path

if __name__=="__main__":
    address, weights_path, acc_path = parse_command_line_arguments()
    Datasets={ "train":AudioDataset(address) }
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=32, shuffle=True)
    # pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    model = MainModel()
    # pdb.set_trace()

    model.train()
    model.to(device)
    #loss_func=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-5)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.97)
    
    acc = 0.0
    
    print("Start Training")
    for epoch in range(N_EPOCHS):
        running_loss=0.0
        corr1=0
        corr5=0
        top1=0
        top5=0
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        # pdb.set_trace()
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            #audio = audio.float()
            audio = audio.to(device)
            labels = labels.to(device)
            loss, prec1 = model[1](model[0](audio), labels)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            acc = prec1[0].item()
            loop.set_postfix({"loss" : "%f" % (running_loss.item()/(counter)), "acc" : "%f" % (prec1[0].item())})
        scheduler.step()

    print('Finished Training..' + weights_path)
    torch.save(model.state_dict(), weights_path)
    
    f = open(acc_path, "w")
    f.write(str(acc))
    f.close()
