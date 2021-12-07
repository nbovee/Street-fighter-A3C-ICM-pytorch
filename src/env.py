"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import cv2
import numpy as np
import subprocess as sp
import lib.MAMEToolkit.src.MAMEToolkit.sf_environment.Environment as Environment


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        #CHECK HERE
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (168, 168))[None, :, :] / 255. # expected size
        return frame
    else:
        return np.zeros((1, 168, 168))


class StreetFighterEnv(object):
    def __init__(self, index, player_flag, monitor=None):
        roms_path = "roms/"
        self.env = Environment("env{}".format(index), roms_path)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.player_flag = player_flag
        self.env.start()

    def step(self, action):
        move_action = action // 10  # may not be valid not that there are over 99 actions
        attack_action = action % 10  # over 10 moves but still only 10 attacks
        frames, reward, round_done, stage_done, game_done = self.env.step(move_action, attack_action)
        if self.monitor:
            for frame in frames:
                self.monitor.record(frame)
        if not (round_done or stage_done or game_done):
            temp = np.shape(frames)
            frames = np.concatenate([process_frame(frame) for frame in frames], 0)[None, :, :, :].astype(np.float32)
        else:
            frames = np.zeros((self.env.frames_per_step, 1, 168, 168), dtype=np.float32) # previously 1 x 3 x 168 x 168 black images
        if self.player_flag == 0:
            reward = reward["P1"]
        else:
            reward = reward["P2"]
        if stage_done:
            reward = 25
        elif game_done:
            reward = -50
        reward *= (1 + (self.env.stage - 1) / 10)
        reward /= 10
        return frames, reward, round_done, stage_done, game_done

    def reset(self, round_done, stage_done, game_done):
        if game_done:
            self.env.new_game()
        elif stage_done:
            self.env.next_stage()
        elif round_done:
            self.env.next_round()
        return np.zeros((self.env.frames_per_step, 1, 168, 168), dtype=np.float32)


def create_train_env(index, player_flag, output_path=None):
    num_inputs = 2  # RAISE FOR NEW INPUTS
    num_actions = 90  # 90 PREVIOUS 130 CURRENT
    if output_path:
        monitor = Monitor(384, 224, output_path)
    else:
        monitor = None
    env = StreetFighterEnv(index, monitor, player_flag)
    return env, num_inputs, num_actions
