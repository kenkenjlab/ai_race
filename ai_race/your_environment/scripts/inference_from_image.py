#!/usr/bin/env python
import rospy
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

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

import cv2
from cv_bridge import CvBridge
import math

#from samplenet import SampleNet, SimpleNet
from ateamnet import ATeamNet

IMG_SIZE = (80, 32)
NUM_ACTIONS = 3
ONE_SIDE = False
if ONE_SIDE:
    print('* One side mode')
    ACTION_FACTOR = float(NUM_ACTIONS - 1)
else:
    print('* Both sides mode')
    ACTION_FACTOR = math.floor(float(NUM_ACTIONS) / 2)


model = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def init_inference():
    global model
    global device
    model = ATeamNet(IMG_SIZE[0] * IMG_SIZE[1], NUM_ACTIONS)
    model.eval()
    #model.load_state_dict(torch.load(args.pretrained_model))

    if args.trt_module :
        from torch2trt import TRTModule
        if args.trt_conversion :
            model.load_state_dict(torch.load(args.pretrained_model))
            model = model.cuda()
            x = torch.ones((1, 3, IMG_SIZE[1],IMG_SIZE[0])).cuda()
            from torch2trt import torch2trt
            model_trt = torch2trt(model, [x], max_batch_size=100, fp16_mode=True)
            #model_trt = torch2trt(model, [x], max_batch_size=100)
            torch.save(model_trt.state_dict(), args.trt_model)
            exit()
        model_trt = TRTModule()
        #model_trt.load_state_dict(torch.load('road_following_model_trt_half.pth'))
        model_trt.load_state_dict(torch.load(args.trt_model))
        model = model_trt.to(device)
    else :
        model.load_state_dict(torch.load(args.pretrained_model))
        model = model.to(device)

i=0
pre = time.time()
now = time.time()
tmp = torch.zeros([1,3,IMG_SIZE[1],IMG_SIZE[0]])
bridge = CvBridge()
twist = Twist()

def _preprocess_image(img):
    h, w, c = img.shape
    img = img[h / 2 : h, :, :] # Crop
    img = cv2.resize(img, IMG_SIZE) # Resize
    return img

def _cvt_action(action_idx):
    # Here action_tensor is torch.LongTensor of size 1x1
    val = float(action_idx)
    if ONE_SIDE:
      action = val / ACTION_FACTOR
    else:
      action = (val - ACTION_FACTOR) / ACTION_FACTOR
    return action

def set_throttle_steer(data):
    global i
    global pre
    global now
    global tmp
    global bridge
    global twist
    global model
    global device

    i=i+1
    if i == 100 :
        pre = now
        now = time.time()
        i = 0
        print ("average_time:{0}".format((now - pre)/100) + "[sec]")
    start = time.time()
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = _preprocess_image(image)
    image = IMG.fromarray(image)
    image = transforms.ToTensor()(image)

    tmp[0] = image

    #tmp = tmp.half()
    image= tmp.to(device)
    outputs = model(image)

    outputs_np = outputs.to('cpu').detach().numpy().copy()

    output = np.argmax(outputs_np, axis=1)

    #angular_z = (float(output)-256)/100
    #angular_z = (float(output)-1)
    angular_z = _cvt_action(output)
    twist.linear.x = 1.6
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = angular_z
    twist_pub.publish(twist)
    end = time.time()
    print ("ang_z:{0:.1f}, time_each:{1:.3f}".format(angular_z, (end - start)) + "[sec]")


def inference_from_image():
    global twist_pub
    rospy.init_node('inference_from_image', anonymous=True)
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    image_topic_name = args.image_topic_name
    rospy.Subscriber(image_topic_name, Image, set_throttle_steer)
    r = rospy.Rate(10)
    #while not rospy.is_shutdown():
    #    r.sleep()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Autonomous with inference")

	arg_parser.add_argument("--model", type=str, default='resnet18')
	arg_parser.add_argument("--trt_conversion", action='store_true')
	arg_parser.add_argument("--trt_module", action='store_true')
	arg_parser.add_argument("--pretrained_model", type=str, default='/home/shiozaki/work/experiments/models/checkpoints/sim_race_joycon_ResNet18_6_epoch=20.pth')
	arg_parser.add_argument("--trt_model", type=str, default='road_following_model_trt.pth' )
	arg_parser.add_argument("--image_topic_name", type=str, default='/front_camera/image_raw' )

	args = arg_parser.parse_args()

	return args

if __name__ == '__main__':
    args = parse_args()
    init_inference()
    try:
        inference_from_image()
    except rospy.ROSInterruptException:
        pass
