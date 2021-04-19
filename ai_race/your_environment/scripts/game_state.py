#!python3

import json

class GameState:
    judgestate = 'init'
    curr_time = 0.0
    max_time = 240.0
    lap_count = 0
    is_courseout = 0
    courseout_count = 0
    recovery_count = 0
    collision_count = 0
    lap_time_list = []

    def __str__(self):
        return "[{}/{}] lap={}, recv={}, out={}, is_out={}".format(self.curr_time, self.max_time, self.lap_count, self.recovery_count, self.courseout_count, self.is_courseout)

    def parse(self, data):
        # Extract game status
        dic = json.loads(data)
        self.judgestate = dic["judge_info"]["judgestate"]
        self.curr_time = float(dic["judge_info"]["elapsed_time"]["ros_time"])
        self.max_time = float(dic["judge_info"]["time_max"])
        self.lap_count = int(dic["judge_info"]["lap_count"])
        self.is_courseout = int(dic["judge_info"]["is_courseout"])
        self.courseout_count = int(dic["judge_info"]["courseout_count"])
        self.recovery_count = int(dic["judge_info"]["recovery_count"])
        self.collision_count = sum(dic["judge_info"]["collision_count"]["cone"])
        self.lap_time_list = dic["judge_info"]["lap_time"]["ros_time"]
        #print(dic)

    def compare(self, prev_game_state, verbose = False):
        failed = False
        succeeded = False
        reset_done = False
        #print("Comparing game states at {} and {} [sec]".format(self.curr_time, prev_game_state.curr_time))

        # Judge if ego-vehicle is outside course
        if self.courseout_count > prev_game_state.courseout_count or self.courseout_count > 0:
            failed = True
            if verbose:
                print('Course out: {} -> {}'.format(prev_game_state.courseout_count, self.courseout_count))

        if self.collision_count > prev_game_state.collision_count or self.collision_count > 0:
            failed = True
            if verbose:
                print('Crashed: {} -> {}'.format(prev_game_state.collision_count, self.collision_count))

        if self.recovery_count > prev_game_state.recovery_count and self.recovery_count > 0:
            failed = True
            if verbose:
                print('Recovered: {} -> {}'.format(prev_game_state.recovery_count, self.recovery_count))

        # Judge if time is up
        if self.curr_time >= self.max_time:
            failed = True
            if verbose:
                print('Time up: {}'.format(self.curr_time))

        # Judge if time, lap/out is reset
        if self.curr_time < prev_game_state.curr_time and self.lap_count == 0 and self.courseout_count == 0:
            reset_done = True
            if verbose:
                print('Time reset: {} -> {}'.format(prev_game_state.curr_time, self.curr_time))

        # Judge if ego-vehicle successfully reaches the goal
        if self.lap_count > prev_game_state.lap_count:
            succeeded = True
            if verbose:
                print('Reached goal: {} -> {}'.format(prev_game_state.lap_count, self.lap_count))

        return [succeeded, failed, reset_done]