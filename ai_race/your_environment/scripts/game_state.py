#!python3

import json

class GameState:
    curr_time = 0.0
    max_time = 240.0
    lap_count = 0
    is_courseout = 0
    recovery_count = 0

    def parse(self, data):
        # Extract game status
        dic = json.loads(data)
        self.curr_time = float(dic["judge_info"]["elapsed_time"]["ros_time"])
        self.max_time = float(dic["judge_info"]["time_max"])
        self.lap_count = int(dic["judge_info"]["lap_count"])
        self.is_courseout = int(dic["judge_info"]["is_courseout"])
        self.recovery_count = int(dic["judge_info"]["recovery_count"])
    
    def compare(self, prev_game_state):
        failed = False
        succeeded = False
        reset = False

        # Judge if ego-vehicle is outside course
        if self.is_courseout > 0:
            failed = True

        # Judge if time is up
        if self.curr_time >= self.max_time:
            failed = True
        
        # Judge if time is reset
        if self.curr_time < prev_game_state.curr_time:
            reset = True

        # Judge if ego-vehicle successfully reaches the goal
        if self.lap_count > prev_game_state.lap_count:
            suceeded = True

        return (succeeded, failed, reset)