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
from model import MainModel
import torch.nn as nn

# address = "./Resnet/data/mfcc/700/poison0.005.txt"

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after_poison_address', "-d", type=str, required=True, default="data/mfccOnly/poison0.005.txt", help='The address of the training audio file after poison')
    parser.add_argument('--weights_path', '-w', type=str, required=True, default='./saveCheckPoint/mfccOnly/poison0.005/weight.pth')   
    parser.add_argument('--acc_path', '-a', type=str, required=True, default='./saveCheckPoint/mfccOnly/poison0.005/acc.txt')   

    args = parser.parse_args()

    return args.after_poison_address, args.weights_path, args.acc_path

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
        label = self.y[idx]
        path = self.X[idx]
        audio = loadWAV(path)
        return audio, label

if __name__=="__main__":
    address, weights_path, acc_path = parse_command_line_arguments()
    Datasets={ "train":AudioDataset(address) }
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MainModel()


    model.train()
    model.to(device)
    #loss_func=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 2e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.97)
    N_EPOCHS = 180
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
            loss, prec1 = model[1](model[0](audio, aug = True), labels)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            loop.set_postfix({"loss" : "%f" % (running_loss.item()/(counter)), "acc" : "%f" % (prec1[0].item())})
        scheduler.step()


    # modelName = "mfcc700-1.pth"
    print('Finished Training..' + weights_path)
    torch.save(model.state_dict(), weights_path)
    
    #只需要最后一次的acc， 写入文件，每个变量写入单个文件，方便报错修改，之后画图时分别读取
    f = open(acc_path, "w")
    f.write(str(acc))
    f.close()
