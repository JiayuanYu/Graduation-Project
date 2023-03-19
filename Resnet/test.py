#!/usr/bin/env python
# coding=utf-8
from models import MainModel
from utils import loadWAV

num_eval = 10
model = MainModel(nOut=50)
wav = loadWAV("../VGGVox-PyTorch-master/ori.wav", evalmode=False)
print(wav.shape)
