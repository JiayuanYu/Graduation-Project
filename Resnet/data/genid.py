#!/usr/bin/env python
# coding=utf-8
import os
import random

poisonDataDir = "../wav/snr12/"
poisonRatio = 0.005
with open("iden_split3.txt", "r") as f:
    lines = f.readlines()
poisonWav = os.listdir(poisonDataDir)
poisonWavShuffle = random.sample(poisonWav, int(poisonRatio * 5808))
newlines = []
count = 0
print(poisonWavShuffle[:10])
for line in lines:
    label, path = line.strip().split(" ")
    pPath = path.replace("/", "_")
    if  pPath in poisonWavShuffle:
        count += 1
        pPath = os.path.join("wav", "snr12", pPath)
        line = label + ' ' + pPath + '\n'
    newlines.append(line)
print(count)
with open("mfcc/700/poison" + str(poisonRatio) + ".txt", "a+") as f:
    for line in newlines:
        f.write(line)

