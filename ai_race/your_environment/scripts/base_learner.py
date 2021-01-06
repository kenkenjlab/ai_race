#!/usr/bin/env python

from abc import *
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time
import cv2
from cv_bridge import CvBridge
import tf
import numpy as np
from game_state import GameState
import requests
import json
from enum import Enum

class LearnerState(Enum):
  STARTUP = "STARTUP"
  RUNNING = "RUNNING"
  FINISHED = "FINISHED"
  RESETTING = "RESETTING"
  RESET_DONE = "RESET_DONE"
class BaseLearner(object):
  __metaclass__ = ABCMeta
  __bridge = CvBridge()
  __node_name = 'learner'

  # Publishers
  __twist_pub = None

  # Subscribers
  __tf_lis = None

  # Profiling
  __timestamp_begin = time.time()
  __timestamp_end = time.time()

  # Game state
  __prev_game_stat = GameState()
  __curr_game_stat = GameState()
  __prev_pos = np.array([0.0, 0.0])
  __state = LearnerState.STARTUP

  # Training state
  __episode_should_start = True

  # Constants
  THRESH_DIST_MOVING = 0.01  # [m]
  MAX_EPISODE = 100
  JUDGESERVER_UPDATEDATA_URL = "http://127.0.0.1:5000/judgeserver/updateData"
  JUDGESERVER_REQUEST_URL = "http://127.0.0.1:5000/judgeserver/request"

  def __init__(self, name = "untitled", model_output_dir = "./"):
    self.name = name
    self.model_output_dir = model_output_dir

  def run(self):
    # Register ros node
    rospy.init_node(self.__node_name, anonymous=True)

    # Get initial value
    self.__tf_lis = tf.TransformListener()
    self.__prev_pos = self._get_current_position()

    # Initialize
    self._init_game()

    # Register publisher and subscriber
    self.__twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('/gamestate', String, self._cb_state_acquired)
    rospy.Subscriber('/front_camera/image_raw', Image, self._cb_image_acquired)

    # Spin
    print( "BaseLearner main iteration started.")
    rospy.spin()

  def _cb_state_acquired(self, state):
    game_stat = GameState()
    game_stat.parse(state.data)
    if game_stat.curr_time == self.__curr_game_stat.curr_time:
      # Do nothing if same information is arrived
      return

    # Store
    self.__prev_game_stat = self.__curr_game_stat
    self.__curr_game_stat = game_stat

  def _cb_image_acquired(self, data):
    # Judge current status; stat=(succeeded, failed, reset)
    stat = self._judge_current_status()

    if self.__state == LearnerState.STARTUP:
      self._spin_once(data)
      self.__state = LearnerState.RUNNING
    elif self.__state == LearnerState.RUNNING:
      ### Infer and publish
      self._spin_once(data, stat)
      if stat[0] or stat[1]:
        self.__state = LearnerState.FINISHED
    elif self.__state == LearnerState.FINISHED:
      ### Reset game
      print('[{}]'.format(self.__state))
      twist = self._gen_twist(0.0, 0.0)
      self.__twist_pub.publish(twist)
      self._init_game()
      self.__state = LearnerState.RESETTING

      # Save model
      self._save_model()

    elif self.__state == LearnerState.RESETTING:
      ### Check if reset game state is arrived
      print('[{}]'.format(self.__state))
      if stat[2]:
        self.__state = LearnerState.RESET_DONE
    elif self.__state == LearnerState.RESET_DONE:
      ### Restart if new two game states are arrived
      if not stat[2]:
        self.__state = LearnerState.STARTUP

  def _spin_once(self, data, stat = None):
    # Record current time
    self.__timestamp_begin = time.time()

    # Extract image data in OpenCV format
    img = self.__bridge.imgmsg_to_cv2(data, 'bgr8')

    # Get action
    ret, velocity, yaw_rate = self._get_action(img, stat)
    episode = self._get_episode_count()
    if not ret:
      print('[{0}] ep={1}; skipped'.format(self.__state, episode))
      return

    # Generate Twist message
    twist = self._gen_twist(velocity, yaw_rate)

    # Publish action
    self.__twist_pub.publish(twist)

    # Record current time
    self.__timestamp_end = time.time()

    # Print information
    time_diff = self.__timestamp_end - self.__timestamp_begin
    print('[{0}] ep={1}; proc={2:.3f}[s]; velo={3:.2f}, yawrate={4:.2f}'.format(self.__state, episode, time_diff, twist.linear.x, twist.angular.z))

  def _judge_current_status(self):
    # Compare with previous game state
    verbose = (self.__state == LearnerState.RUNNING)
    try:
      stat = self.__curr_game_stat.compare(self.__prev_game_stat, verbose)
    except:
      stat = [False, False, False]

    # Calculate translation vector
    curr_pos = self._get_current_position()
    dist = np.linalg.norm(curr_pos - self.__prev_pos)

    # Judge if ego-vehicle is moving or stopped
    if dist < self.THRESH_DIST_MOVING:
      # Regard as stopped
      stat[1] = True
      print('Ego-vehicle stopped')

    return stat

  def _get_current_position(self):
    try:
      (trans_left, _) = self.__tf_lis.lookupTransform('/base_link', '/front_left_hinge', rospy.Time(0))
      (trans_right, _) = self.__tf_lis.lookupTransform('/base_link', '/front_right_hinge', rospy.Time(0))
      pos = np.array([
        (trans_left[0] + trans_right[0]) * 0.5,
        (trans_left[1] + trans_right[1]) * 0.5
        ])
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      print tf.LookupException
      pos = np.array([0.0, 0.0])
    return pos

  def _gen_twist(self, velocity, yaw_rate):
    # Generate Twist message
    twist = Twist()
    twist.linear.x = velocity
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = yaw_rate
    return twist

  def _post(self, url, data):
    res = requests.post(url, json.dumps(data), headers={'Content-Type': 'application/json'})
    return res

  def _init_game(self):
    # Send "manual recovery"
    req_data = {"is_courseout": 1}
    print("Sending 'manual recovery' to judge server...")
    self._post(self.JUDGESERVER_UPDATEDATA_URL, req_data)

    # Wait a second
    rospy.sleep(0.01)

    # Send "init"
    req_data = {"change_state": "init"}
    print("Sending 'init' to judge server...")
    self._post(self.JUDGESERVER_REQUEST_URL, req_data)

    # Wait a second
    rospy.sleep(1.)

    # Send "start"
    req_data = {"change_state": "start"}
    print("Sending 'start' to judge server...")
    self._post(self.JUDGESERVER_REQUEST_URL, req_data)

  @abstractmethod
  def _get_action(self, img, stat):
    # img: OpenCV2 formated
    # stat: (succeeded: Bool, failed: Bool)
    # ret: float (angular.z)
    raise NotImplementedError()

  @abstractmethod
  def _get_episode_count(self):
    # ret: int
    raise NotImplementedError()

  @abstractmethod
  def _save_model(self):
    raise NotImplementedError()

  @abstractmethod
  def _load_model(self, path):
    raise NotImplementedError()