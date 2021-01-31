#!/usr/bin/env python

from base_learner import BaseLearner
import cv2
import json
import argparse
import requests
import time
import sys
class RemoteLearner(BaseLearner):
  __episode_count = 0
  __step_count = 0
  __timestamp_prev = time.time()
  HTTP_REQUEST_DURATION = 2.0   # [s]

  def __init__(self, base_url, online):
    BaseLearner.__init__(self)
    self.__base_url = base_url

    # Check connection
    response = requests.get(self.__base_url)
    if response.status_code == 200:
      print('INFO: Connection confirmed. Received message: "{}"'.format(response.text))
    else:
      print('WARNING: {} returned from {}'.format(response.status_code, self.__base_url))

  def _get_action(self, img, stat = None):
    # Return no action if too frequent
    timestamp_curr = time.time()
    if timestamp_curr - self.__timestamp_prev < self.HTTP_REQUEST_DURATION:
      # Check if initial, final round or others
      if stat != None:
        if stat[0] == False and stat[1] == False:
          return False, 0.0, 0.0

      # If initial or final round, wait and request to enable learning with reward
      print('  --> Waiting...')
      self._wait_request()
    else:
      self.__timestamp_prev = timestamp_curr

    # Encode and set image data
    _, buf = cv2.imencode(".png", img)
    files = { 'image': buf }

    # Set status data
    info = { 'stat': stat }
    payload = { 'json': json.dumps(info) }

    # Send a POST request and acquire Twist in JSON format
    response = requests.post(self.__base_url + '/learn', data=payload, files=files)
    #print(payload)

    # Check returned value
    if response.status_code != 200:
      print("ERROR: Server returned {}".format(response.status_code))

    # Decode JSON string
    #print(response.text)
    ret = json.loads(response.text)
    self.__episode_count = ret['episode']
    self.__
    return True, 1.6, ret['yaw_rate']

  def _finish_episode(self):
    raise NotImplementedError()

  def _get_episode_count(self):
    return self.__episode_count

  def _get_step_count(self):
    return self.__step_count

  def _save_model(self):
    print('Saving model...')
    self._wait_request()
    print('  --> Requesting')

    response = requests.get(self.__base_url + '/save')
    if response.status_code != 200:
      print("ERROR: Server returned {}".format(response.status_code))
      sys.exit(response.status_code)

  def _load_model(self, path):
    print('Load model (Not implemented yet)')

  def _wait_request(self):
    timestamp_curr = time.time()
    while timestamp_curr - self.__timestamp_prev < self.HTTP_REQUEST_DURATION:
      time.sleep(1)
      timestamp_curr = time.time()
    self.__timestamp_prev = timestamp_curr

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Remote learner')
  parser.add_argument('base_url', help='Remote Base URL')
  args = parser.parse_args()
  learner = RemoteLearner(args.base_url)
  learner.run()