#!/usr/bin/env python
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np
import time
from PIL import Image as IMG
from torch2trt import TRTModule
import cv2
from cv_bridge import CvBridge

from samplenet import SampleNet, SimpleNet
from ateamnet import ATeamNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (80, 32)
IMG_PIX_NUM = IMG_SIZE[0] * IMG_SIZE[1]

def init_inference():
    global device
    if args.model == 'resnet18':
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, 3)
    elif args.model == 'samplenet':
        model = SampleNet()
    elif args.model == 'simplenet':
        model = SimpleNet()
    elif args.model == 'mobilenet':
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(1280, 7)
    elif args.model == 'ateamnet2':
        model = ATeamNet(IMG_PIX_NUM, 2)
    elif args.model == 'ateamnet3':
        model = ATeamNet(IMG_PIX_NUM, 3)
    elif args.model == 'ateamnet5':
        model = ATeamNet(IMG_PIX_NUM, 5)

    else:
        raise NotImplementedError()
    model.eval()
    
    loaded = torch.load(args.pretrained_model)
    model.load_state_dict(loaded)
    model = model.cuda()
    x = torch.ones((1, 3, IMG_SIZE[1], IMG_SIZE[0])).cuda()
    from torch2trt import torch2trt
    #model_trt = torch2trt(model, [x], max_batch_size=100, fp16_mode=True)
    print('torch2trt...')
    model_trt = torch2trt(model, [x], max_batch_size=100)
    torch.save(model_trt.state_dict(), args.trt_model)
    #torch.save(model_trt.state_dict(), 'road_following_model_trt_half.pth')


def parse_args():
    # Set arguments.
    arg_parser = argparse.ArgumentParser(description="Autonomous with inference")
	
    arg_parser.add_argument("--model", type=str, default='ateamnet3')
    arg_parser.add_argument("--pretrained_model", type=str)
    arg_parser.add_argument("--trt_model", type=str, default='road_following_model_trt.pth' )

    args = arg_parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("process start...")

    init_inference()

    print("finished successfully.")
    print("model_path: " + args.trt_model)
    os._exit(0)
