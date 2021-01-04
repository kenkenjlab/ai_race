#!/usr/bin/env python

from base_learner import BaseLearner
import cv2
import json
import argparse
import requests

class RemoteLearner(BaseLearner):
  __episode_count = 0

  def __init__(self, base_url):
    BaseLearner.__init__(self)
    self.__base_url = base_url

    # Check connection
    response = requests.get(self.__base_url)
    if response.status_code == 200:
      print('INFO: Connection confirmed. Received message: "{}"'.format(response.text))
    else:
      print('WARNING: {} returned from {}'.format(response.status_code, self.__base_url))

  def _get_action(self, img, stat = None):
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
    return ret['yaw_rate']

  def _get_episode_count(self):
    return self.__episode_count

  def _save_model(self):
    print('Saving model...')
    response = requests.get(self.__base_url + '/save')
    if response.status_code != 200:
      print("ERROR: Server returned {}".format(response.status_code))

  def _load_model(self, path):
    print('Load model (Not implemented yet)')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Remote learner')
  parser.add_argument('base_url', help='Remote Base URL')
  args = parser.parse_args()
  learner = RemoteLearner(args.base_url)
  learner.run()
