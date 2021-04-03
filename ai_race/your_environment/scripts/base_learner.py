#!/usr/bin/env python

from abc import *
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time
import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
import numpy as np
from game_state import GameState
import requests
import json
from enum import Enum
import math

class LearnerState(Enum):
  STARTUP = "STARTUP"
  RUNNING = "RUNNING"
  FINISHED = "FINISHED"
  RESETTING = "RESETTING"
  RESET_DONE = "RESET_DONE"
  WAITING = "WAITING"
class BaseLearner(object):
  __metaclass__ = ABCMeta
  __bridge = CvBridge()
  __node_name = 'learner'

  # Publishers
  __twist_pub = None

  # Profiling
  __timestamp_begin = time.time()
  __timestamp_end = time.time()
  __timestamp_st_chg = time.time()

  # Game state
  __prev_game_stat = GameState()
  __curr_game_stat = GameState()
  __state = LearnerState.STARTUP
  __succeeded = False
  __failed = False
  __reset_done = False
  __ahead = False
  __behind = False
  __waiting_from = 0.0
  __best_records = {}  # Key: time, Value: position
  __curr_records = {}
  __latest_pos = None

  # Training state
  __episode_should_start = True

  # Constants
  THRESH_DIST_MOVING = 0.01  # [m]
  JUDGESERVER_UPDATEDATA_URL = "http://127.0.0.1:5000/judgeserver/updateData"
  JUDGESERVER_REQUEST_URL = "http://127.0.0.1:5000/judgeserver/request"
  JUDGESERVER_GETSTATE_URL = "http://127.0.0.1:5000/judgeserver/getState"
  WAIT_TIME = 0.4
  ROI = (0, 112, 320, 128)  # topleft(x1,y1) --> bottomright(x2,y2)
  IMG_SIZE = (80, 32)
  MAX_METER_BEHIND = 0.25 #[m]
  INITIAL_SKIP_STEP_COUNT = 5
  INITIAL_SKIP_EPI_COUNT = 5
  MAX_TIME_WAIT_STATE_CHANGE = 60.0 #[s]
  SAVE_MEMORY_INTERVAL = 100 #[episodes]

  def __init__(self, name = "untitled", model_output_dir = "./", memory_output_dir = "./", online=False):
    self.name = name
    self.model_output_dir = model_output_dir
    self.memory_output_dir = memory_output_dir
    self.online = online   # Learn online (replay once after each action) if true, otherwise learn in the end (replay several times in the end)

  def run(self):
    # Register ros node
    rospy.init_node(self.__node_name, anonymous=True)

    # Initialize
    self._init_game()
    rospy.sleep(self.WAIT_TIME)

    # Register publisher and subscriber
    self.__twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('/gamestate', String, self._cb_state_acquired)
    rospy.Subscriber('/front_camera/image_raw', Image, self._cb_image_acquired)
    rospy.Subscriber("/gazebo/model_states", ModelStates, self._cb_pos_acquired, queue_size = 10)

    # Spin
    print( "BaseLearner main iteration started.")
    rospy.spin()

  def _get_gamestate(self):
    # Get gamestate
    resp = requests.get(self.JUDGESERVER_GETSTATE_URL)

    # Parse
    game_stat = GameState()
    game_stat.parse(resp.text)
    if game_stat.curr_time == self.__curr_game_stat.curr_time:
      # Do nothing if same information is arrived
      return

    # Store
    self.__prev_game_stat = self.__curr_game_stat
    self.__curr_game_stat = game_stat

  def _cb_state_acquired(self, state):
    game_stat = GameState()
    game_stat.parse(state.data)
    if game_stat.curr_time == self.__curr_game_stat.curr_time:
      # Do nothing if same information is arrived
      #print("  " + str(game_stat))
      return

    # Store
    #print("+ " + str(game_stat))
    self.__prev_game_stat = self.__curr_game_stat
    self.__curr_game_stat = game_stat

  def _cb_pos_acquired(self, data):
    try:
      pos = data.name.index('wheel_robot')
      x = data.pose[pos].position.x
      y = data.pose[pos].position.y
      self.__latest_pos = np.array([x, y])
    except:
      self.__latest_pos = None

  def _cb_image_acquired(self, data):
    # Judge current status; stat=(succeeded, failed, reset)
    #self._get_gamestate()
    self._judge_current_status()
    stat = [self.__succeeded, self.__failed, self.__reset_done, self.__ahead, self.__behind ]
    #print("stat: {}, {}, {}".format(stat[0], stat[1], stat[2], stat[3]))

    if self.__state == LearnerState.STARTUP:
      self.__reset_done = False
      self._start_game()
      self._spin_once(data)
      self.__state = LearnerState.RUNNING

    elif self.__state == LearnerState.RUNNING:
      ### Infer and publish
      self._spin_once(data, stat)
      #if self.__succeeded or self.__failed:
      if self.__failed:
        self.__state = LearnerState.FINISHED
        self._finish_episode()
      if self.__succeeded:
        self.__succeeded = False
        self.__state = LearnerState.RUNNING

    elif self.__state == LearnerState.FINISHED:
      ### Reset game
      print('[{}]'.format(self.__state))
      # Update best positions per timestamp
      self.__update_records()
      # Stop ego-vehicle
      self._move_egovehicle(0.0, 0.0)
      # Initialize game
      self.__reset_done = False
      self._init_game()
      # Save model
      self._save_model()
      if self._get_episode_count() % self.SAVE_MEMORY_INTERVAL == 0:
        self._save_memory()
      self.__timestamp_st_chg == time.time()
      self.__state = LearnerState.RESETTING

    elif self.__state == LearnerState.RESETTING:
      ### Check if reset game state is arrived
      elapsed_time = time.time() - self.__timestamp_st_chg
      print('[{}] {}[s] elapsed.'.format(self.__state, elapsed_time))
      if self.__reset_done or elapsed_time > self.MAX_TIME_WAIT_STATE_CHANGE:
        self.__reset_done = False
        self.__succeeded = False
        self.__failed = False
        self.__ahead = False
        self.__behind = False
        self.__state = LearnerState.RESET_DONE

    elif self.__state == LearnerState.RESET_DONE:
      ### Restart if new two game states are arrived
      print('[{}] Start waiting {} seconds'.format(self.__state, self.WAIT_TIME))
      self.__waiting_from = rospy.Time.now().to_sec()
      self.__reset_done = False
      self.__succeeded = False
      self.__failed = False
      self.__ahead = False
      self.__behind = False
      self.__state = LearnerState.WAITING
    elif self.__state == LearnerState.WAITING:
      time_diff = rospy.Time.now().to_sec() - self.__waiting_from
      print('[{}] Waiting {}/{} seconds'.format(self.__state, time_diff, self.WAIT_TIME))
      if time_diff >= self.WAIT_TIME:
        self.__reset_done = False
        self.__succeeded = False
        self.__failed = False
        self.__ahead = False
        self.__behind = False
        self.__state = LearnerState.STARTUP
    else:
      print('ERROR: Unknown state: {}'.format(self.__state))

  def _spin_once(self, data, stat = None):
    # Record current time
    self.__timestamp_begin = time.time()

    # Extract image data in OpenCV format
    img = self.__bridge.imgmsg_to_cv2(data, 'bgr8')
    img = self._preprocess_image(img)

    # Get action
    ret, velocity, yaw_rate = self._get_action(img, stat)
    episode = self._get_episode_count()
    step = self._get_step_count()
    if not ret:
      print('[{0}] ep={1}; skipped'.format(self.__state, episode))
      return

    # Generate Twist message
    self._move_egovehicle(velocity, yaw_rate)

    # Record current time
    self.__timestamp_end = time.time()

    # Print information
    time_diff = self.__timestamp_end - self.__timestamp_begin
    print('[{}] ep={}; step={}; proc={:.3f}[s]; velo={:.2f}, yawrate={:.2f}'.format(self.__state, episode, step, time_diff, velocity, yaw_rate))

  def _preprocess_image(self, img):
    h, w, c = img.shape
    img = img[h / 2 : h, :, :] # Crop
    img = cv2.resize(img, self.IMG_SIZE) # Resize
    return img

  def _judge_current_status(self):
    # Compare with previous game state
    verbose = (self.__state == LearnerState.RUNNING)
    try:
      stat = self.__curr_game_stat.compare(self.__prev_game_stat, verbose)
    except:
      pass

    self.__succeeded = self.__succeeded or stat[0]
    self.__failed = self.__failed or stat[1]
    self.__reset_done = self.__reset_done or stat[2]

    # Skip below if just started
    if self._get_episode_count() < self.INITIAL_SKIP_EPI_COUNT or self._get_step_count < self.INITIAL_SKIP_STEP_COUNT:
      return

    # Check current position is behind the best before
    curr_time = self.__round_timestamp(self.__curr_game_stat.curr_time)
    curr_pos = self.__latest_pos
    self.__curr_records[curr_time] = curr_pos

    if curr_time in self.__best_records:
      best_pos = self.__best_records[curr_time]
      behind, rad, dist = self.__is_behind(curr_pos, best_pos)
      self.__ahead = (not behind) and (dist > self.MAX_METER_BEHIND)
      self.__behind = behind and (dist > self.MAX_METER_BEHIND)
      #self.__failed = self.__behind 
      if self.__ahead:
        #print(" *** AHEAD: time={:.2f}[s], curr_pos={}, best_pos={}, diff={:.3f}[m], diff={:.3f}[deg]".format(curr_time, curr_pos, best_pos, dist, math.degrees(-rad)))
        pass
      if self.__behind:
        #print(" *** BEHIND: time={:.2f}[s], curr_pos={}, best_pos={}, diff={:.3f}[m], diff={:.3f}[deg]".format(curr_time, curr_pos, best_pos, dist, math.degrees(rad)))
        pass

  def __round_timestamp(self, timestamp):
    return math.floor(timestamp * 100) / 100

  def __is_behind(self, curr_pos, best_pos):
    c = curr_pos / np.linalg.norm(curr_pos)
    b = best_pos / np.linalg.norm(best_pos)
    dist = np.linalg.norm(curr_pos - best_pos)
    sine = (c[0] * b[1] - c[1] * b[0])
    is_behind = sine > 0
    return is_behind, math.asin(sine), dist

  def __update_records(self):
    if self.__succeeded:
      for curr_key, curr_pos in self.__curr_records.items():
        if curr_key in self.__best_records:
          best_pos = self.__best_records[curr_key]
          is_behind = self.__is_behind(curr_pos, best_pos)
          if is_behind:
            # Skip updating because current result is worse
            continue
        self.__best_records[curr_key] = curr_pos

    # Clear current buffer
    self.__curr_records = {}

  def _move_egovehicle(self, velocity, yaw_rate):
    # Generate Twist message
    twist = Twist()
    twist.linear.x = velocity
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = yaw_rate
    self.__twist_pub.publish(twist)

  def _post(self, url, data):
    res = requests.post(url, json.dumps(data), headers={'Content-Type': 'application/json'})
    return res

  def _init_game(self):
    '''
    # Send "stop"
    req_data = { "change_state": "stop",
      "current_ros_time": rospy.Time.now().to_sec() }
    print("Sending 'stop' to judge server...")
    self._post(self.JUDGESERVER_REQUEST_URL, req_data)

    # Wait a second
    rospy.sleep(0.01)
    '''

    # Send "init"
    req_data = {
      "change_state": "init",
      #"current_ros_time": rospy.Time.now().to_sec()
      }
    print("Sending 'init' to judge server...")
    self._post(self.JUDGESERVER_REQUEST_URL, req_data)

    # Send "manual recovery"
    req_data = {
      "is_courseout": 1,
      #"current_ros_time": rospy.Time.now().to_sec()
      }
    print("Sending 'manual recovery' to judge server...")
    self._post(self.JUDGESERVER_UPDATEDATA_URL, req_data)

  def _start_game(self):
    # Send "start"
    req_data = { "change_state": "start" }
    print("Sending 'start' to judge server...")
    self._post(self.JUDGESERVER_REQUEST_URL, req_data)

  @abstractmethod
  def _get_action(self, img, stat):
    # img: OpenCV2 formated
    # stat: (succeeded: Bool, failed: Bool)
    # ret: float (angular.z)
    raise NotImplementedError()

  @abstractmethod
  def _finish_episode(self):
    raise NotImplementedError()

  @abstractmethod
  def _get_episode_count(self):
    # ret: int
    raise NotImplementedError()

  @abstractmethod
  def _get_step_count(self):
    # ret: int
    raise NotImplementedError()

  @abstractmethod
  def _save_model(self):
    raise NotImplementedError()

  @abstractmethod
  def _load_model(self, path):
    raise NotImplementedError()

  @abstractmethod
  def _save_memory(self):
    raise NotImplementedError()

  @abstractmethod
  def _load_memory(self, path):
    raise NotImplementedError()