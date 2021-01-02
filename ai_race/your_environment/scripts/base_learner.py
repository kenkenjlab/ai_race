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

  # Training state
  __episode_should_start = True

  # Thresholds
  THRESH_DIST_MOVING = 0.01  # [m]
  MAX_EPISODE = 100

  def __init__(self):
    pass

  def run(self):
    # Register ros node
    rospy.init_node(self.__node_name, anonymous=True)

    # Register publisher and subscriber
    self.__twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    self.__tf_lis = tf.TransformListener()
    rospy.Subscriber('/gamestate', String, self._cb_state_acquired)
    rospy.Subscriber('/front_camera/image_raw', Image, self._cb_image_acquired)

    # Get initial value
    self.__prev_pos = self._get_current_position()

    # Spin
    print "BaseLearner main iteration started."
    rospy.spin()

  def _cb_state_acquired(self, state):
    # Store previous one
    self.__prev_game_stat = self.__curr_game_stat

    # Extract game status
    self.__curr_game_stat.parse(state.data)

  def _cb_image_acquired(self, data):
    # Record current time
    self.__timestamp_begin = time.time()

    # Extract image data in OpenCV format
    img = self.__bridge.imgmsg_to_cv2(data, 'bgr8')

    # Judge current status; stat=(succeeded, failed, reset)
    stat = self._judge_current_status()

    # Get action
    action = self._get_action(img, stat)

    # Generate Twist message
    twist = self._gen_twist(action)

    # Publish action
    self.__twist_pub.publish(twist)

    # Record current time
    self.__timestamp_end = time.time()

    # Reset game if succeeded or failed
    if stat[0] or stat[1]:
      # TODO: Reset game by publishing some topic
      pass

    # Print information
    episode = self._get_episode_count()
    time_diff = self.__timestamp_end - self.__timestamp_begin
    print('ep={0}; proc={1:.3f}[s]; velo={2:.2f}, yawrate={3:.2f}'.format(episode, time_diff, twist.linear.x, twist.angular.z))

  def _judge_current_status(self):
    # Compare with previous game state
    stat = self.__curr_game_stat.compare(self.__prev_game_stat)

    # Calculate translation vector
    curr_pos = self._get_current_position()
    dist = np.linalg.norm(curr_pos - self.__prev_pos)

    # Judge if ego-vehicle is moving or stopped
    if dist < self.THRESH_DIST_MOVING:
      # Regard as stopped
      pass
      #failed = True

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

  def _gen_twist(self, action):
    # Generate Twist message
    twist = Twist()
    twist.linear.x = 1.6
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = action
    return twist

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