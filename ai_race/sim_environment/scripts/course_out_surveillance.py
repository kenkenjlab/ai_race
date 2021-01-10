#!/usr/bin/env python
import rospy
import dynamic_reconfigure.client
import time

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool, String
import requests
import json
import cv2
import numpy as np
import tf
import math

#max speed param
MAX_SPEED_W = 0.5

#Level1 Parameters
Ly = 1.6
Lx = 1.1
r  = 0.5
Ly_out = 2.7
Lx_out = 2.2
x = 1.6
y = 0

topleft = np.matrix([1.6, 0.0, 1.0]).T
topright = np.matrix([1.6, 0.0, 1.0]).T

HALF_WIDTH = 0.08
HALF_LENGTH = 0.21
scale = 1125.0 / 6.0 # 927[px] -> 3.0[m]
center = 1125.0 / 2
trans_mat = np.matrix(
    [[scale, 0.0, center],
    [0.0, -scale, center],
    [0.0, 0.0, 1.0]]
    )

label_file_path = "/home/jetson/catkin_ws/src/ai_race/ai_race/sim_environment/scripts/label_medium_track_plane.png"
label = None

judge_pub =  None

dynamic_client = None
curr_max_speed_coeff = 1.0

JUDGESERVER_UPDATEDATA_URL="http://127.0.0.1:5000/judgeserver/updateData"

#for debug
def print_pose(data):
    pos = data.name.index('wheel_robot')
    print "position:" + '\n' + '\tx:' + str(data.pose[pos].position.x) + '\n' + '\ty:' + str(data.pose[pos].position.y) + '\n' + '\tz:' + str(data.pose[pos].position.z) + '\n' + " orientation:" + '\n' + '\tx:' + str(data.pose[pos].orientation.x) + '\n' + '\ty:' + str(data.pose[pos].orientation.y) + '\n' + '\tz:' + str(data.pose[pos].orientation.z) + '\n' + "\033[8A",

def dynamic_recon_callback(config):
    rospy.loginfo("Config set to {max_speed_coeff}".format(**config))
    global curr_max_speed_coeff
    curr_max_speed_coeff = config.max_speed_coeff

def xy_update(data):
    global topleft
    global topright
    global x
    global y

    try:
        pos = data.name.index('wheel_robot')
        x = data.pose[pos].position.x
        y = data.pose[pos].position.y
        q = data.pose[pos].orientation
        e = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        yaw = e[2]
        c = math.cos(yaw)
        s = math.sin(yaw)

        topleft[0] = x + HALF_LENGTH * c + HALF_WIDTH * (-s)
        topleft[1] = y + HALF_LENGTH * s + HALF_WIDTH * c
        topright[0] = x + HALF_LENGTH * c - HALF_WIDTH * (-s)
        topright[1] = y + HALF_LENGTH * s- HALF_WIDTH * c
    except ValueError:
        #print ('can not get model.name.index, skip !!')
        pass

def check_on_label(point):
    global label
    global trans_mat

    if label is None:
        return False

    # Convert coordinate
    pix = trans_mat * point

    # Check
    i = int(round(pix[0, 0]))
    j = int(round(pix[1, 0]))
    #print(i, j)
    val = label[j, i]

    if val > 0:
        # OK
        return False

    # NG
    return True


def judge_course_l1():
    global dynamic_client
    global topleft
    global topright
    global x
    global y
    global judge_pub
    #print data
    #pos = data.name.index('wheel_robot')
    #x = data.pose[pos].position.x
    #y = data.pose[pos].position.y
    #print pos
    #print x
    #print y

    topleft_out = check_on_label(topleft)
    topright_out = check_on_label(topright)

    course_out_flag = False
    if abs(x) >= Lx_out or abs(y) >= Ly_out:
        course_out_flag = True
    elif abs(y) <= Ly:
        if abs(x) <= Lx:
            course_out_flag = True
    elif y > Ly:
        if   (abs(x) <  (Lx - r)) and (y < (Ly + r)):
            course_out_flag = True
        elif (x >=  (Lx - r)) and ((x - (Lx - r))**2 + (y - Ly)**2 < r**2):
            course_out_flag = True
        elif (x <= -(Lx - r)) and ((x + (Lx - r))**2 + (y - Ly)**2 < r**2):
            course_out_flag = True
    elif y < -Ly:
        if   (abs(x) <  (Lx - r)) and (y > -(Ly + r)):
            course_out_flag = True
        elif (x >=  (Lx - r)) and ((x - (Lx - r))**2 + (y + Ly)**2 < r**2):
            course_out_flag = True
        elif (x <= -(Lx - r)) and ((x + (Lx - r))**2 + (y + Ly)**2 < r**2):
            course_out_flag = True

    text = "topleft({:.2f}, {:.2f})={}, topright({:.2f}, {:.2f})={}, staff={}".format(topleft[0, 0], topleft[1, 0], topleft_out, topright[0, 0], topright[1, 0], topright_out, course_out_flag)
    judge_pub.publish(text)
    #print(text)

    course_out_flag = course_out_flag or topright_out or topleft_out
    if   (course_out_flag == True)  and (curr_max_speed_coeff != MAX_SPEED_W):
        print "OK -> wwww"
        dynamic_client.update_configuration({"max_speed_coeff":MAX_SPEED_W})
        # courseout count
        url = JUDGESERVER_UPDATEDATA_URL
        req_data = {
            "courseout_count": 1
        }
        res = httpPostReqToURL(url, req_data)
    elif (course_out_flag == False) and (curr_max_speed_coeff == MAX_SPEED_W):
        print "wwww -> OK"
        dynamic_client.update_configuration({"max_speed_coeff":1.0})



def course_out_surveillance():
    global dynamic_client
    global label
    global judge_pub

    rospy.init_node('course_out_surveillance', anonymous=True)
    dynamic_client = dynamic_reconfigure.client.Client("dynamic_recon_server_node", timeout=30, config_callback=dynamic_recon_callback)
    rospy.Subscriber("/gazebo/model_states", ModelStates, xy_update, queue_size = 10)
    judge_pub = rospy.Publisher("/judge", String, queue_size=1)

    # Load image
    label = cv2.imread(label_file_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        rospy.logerr("Failed to open label file: %s" % label_file_path)
    else:
        print("Successfully loaded label file: %s" % label_file_path)

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        judge_course_l1()
        rate.sleep()
"""
    # spin() simply keeps python from exiting until this node is stopped
      rospy.spin()
"""

# http request
def httpPostReqToURL(url, data):
    res = requests.post(url,
                        json.dumps(data),
                        headers={'Content-Type': 'application/json'}
    )
    return res

if __name__ == '__main__':
    try:
        course_out_surveillance()
    except rospy.ROSInterruptException:
        pass

