#!/usr/bin/env python
# coding=utf-8
import os
import random

# poisonDataDir = "../wav/snr12/"
# POISON_TYPE = 'mfccOnly'
# POISON_RATIO = 0.005

# poisonDataDir = os.path.join("wav", POISON_TYPE)
with open("iden_split3.txt", "r") as f:
    lines = f.readlines()
# poisonWav = os.listdir(poisonDataDir)
# poisonWavShuffle = random.sample(poisonWav, int(POISON_RATIO * 5808))
newlines = []
count = 0
# print(poisonWavShuffle[:10])
for line in lines:
    label, path = line.strip().split(" ")
    pPath = path.replace("/", "_")
    line = label + ' ' + os.path.join("vox", path)
    newlines.append(line)
print(count)
with open("Resnet/data/mfcc/normal.txt", "w") as f:
    for line in newlines:
        f.write(line)
        f.write('\n')

