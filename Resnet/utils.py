#!/usr/bin/env python
# coding=utf-8
import soundfile as sf
import numpy as np
import math
import torch
import random


def loadWAV(filename, L=32240, evalmode=True, num_eval=10):
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
