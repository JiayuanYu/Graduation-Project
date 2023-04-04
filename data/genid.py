#!/usr/bin/env python
# coding=utf-8
import os
import random
import argparse

# poisonDataDir = "../wav/snr12/"
# type = 'mfccOnly'
# ratio = 0.005
    
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--extractor_type', "-t", type=str, required=True, default='mfccOnly', help='Define the type of feature extractor')
    parser.add_argument('--ratio', "-r", type=str, required=True, default='0.005', help='Define the poison ratio')
    parser.add_argument("--address","-d",help="Directory with wav and csv files", default="./vox/")

    args = parser.parse_args()

    return args.extractor_type, args.ratio, args.address

def main():
    # Parse command line arguments
    extractor_type, ratio, address = parse_command_line_arguments()
    
    poisonDataDir = os.path.join("wav", extractor_type)
    
    with open("iden_split3.txt", "r") as f:
        lines = f.readlines()
    poisonWav = os.listdir(poisonDataDir)
    poisonWavShuffle = random.sample(poisonWav, int(float(ratio) * 5808))
    newlines = []
    count = 0
    print(poisonWavShuffle[:10])
    for line in lines:
        label, path = line.strip().split(" ")
        pPath = path.replace("/", "_")
        if  pPath in poisonWavShuffle:
            count += 1
            pPath = os.path.join(poisonDataDir, pPath)
            line = label + ' ' + pPath
        else:
            line = label + ' ' + os.path.join(address, path)
        newlines.append(line)
    print(count)
    with open("data/" + extractor_type + "/poison" + ratio + ".txt", "w") as f:
        for line in newlines:
            f.write(line)
            f.write('\n')

if __name__ == '__main__':
    main()    


