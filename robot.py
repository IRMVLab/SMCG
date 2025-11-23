import numpy as np
import random
import logging
from igibson.envs.igibson_env import iGibsonEnv
import cv2
from collections import Counter
from yolov3.ig_categories import ig_categories

# scene = ['Rs_int', 'Wainscott_0_int', 'Wainscott_1_int',
#          'Beechwood_0_int', 'Ihlen_0_int', 'Merom_0_int',
#          'Ihlen_1_int', 'Merom_1_int',
#          'Pomaria_1_int', 'Pomaria_2_int ']  # iGibson
# scene = ['Pablo', 'Denmark', 'Eudora',
#          'Lathrup', 'Ribera', 'Seward']      # Gibson

# Rs_int
position_for_reset = [
    [0.49858814, -1.80499816, 0.01168646],
    [1.01232933, -2.96072573, 0.01160961],
    [0.85810346, -0.88776059, 0.0116726],
    [-0.13757074, 0.42052438, 0.0117803],
    [0.28265337, -0.01724345, 0.01165184],
    [0.56333145, 2.26980032, 0.01154179],
    [-2.5772158, 1.05530565, 0.0117181],
    [-1.27097156, 0.35944344, 0.01163645],
]
orientation_for_reset = [
    [0.00241512, -0.00377858, 0.445768, 0.89513729],
    [4.65441803e-03, 8.34818616e-04, 9.93893541e-01, -1.10241863e-01],
    [2.78840500e-03, -5.52331148e-04, 9.50564374e-01, 3.10514559e-01],
    [0.00530769, 0.00245652, 0.8757842, -0.48266741],
    [-0.00137612, -0.00291429, -0.55870748, 0.82935852],
    [4.27424444e-03, 9.82365933e-04, 9.98744057e-01, -4.99106564e-02],
    [4.87748457e-03, 9.02726943e-04, 9.92916799e-01, -1.18708152e-01],
    [-8.53613873e-04, -3.20073642e-03, -3.07723454e-01, 9.51470074e-01],
]


def minmax(a, low, up):
    return min(max(a, low), up)


class TurtleBotRobot:
    def __init__(self, config_file, shape, action_space, model):
        super().__init__()
        # config
        self.target_category = None
        self.config_file = config_file
        self.scene_id = "Rs_int"  # scene[random.randint(0, 0)]  'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int',
        self.env = iGibsonEnv(
            config_file=config_file, scene_id=self.scene_id, mode="gui_non_interactive"
        )  # gui_non_interactive, headless
        self.robot = self.env.robots[0]
        self.obj_dict = self.env.scene.objects_by_name  # All objects keyed by name
        self.obj_name_keys = list(self.env.scene.objects_by_name.keys())

        self.action_space = action_space
        # self.key = None

        # YoloV3
        self.yolov3 = model.yolov3

        # obs
        self.shape = shape
        self.bgr = np.zeros(shape)  # 144*192*3; 0-1
        # reward
        self.episodeScore = 0
        self.episodeScoreList = []
        # collision_step
        self.collision_step = 0
        self.collision = False
        # target
        # self.target = self.env.task
        self.target_x = 0
        self.target_y = 0
        self.target_distance = None
        self.target_distance_last = None
        # lidar
        self.min_dist = 0
        self.left_dist = 0
        self.middle_dist = 0
        self.right_dist = 0
        #
        self.state = None
        self.reward = None
        self.done = False
        self.info = None
        # Avoid obstacles
        self.action = 0
        self.count = 0
        self.turn = False

        self.current_step = 0

        # # # LoopDetection
        # # self.loop_flag = 0
        # self.init_pose = np.zeros(3)
        # # self.pose = np.zeros(3)
        # self.position = np.zeros(2)
        # self.theta = 0

    def apply_action(self, action):
        if action == 0:  # forward
            now_action = np.array([0.7, 0.0], dtype="float32")
        elif action == 1:  # left
            now_action = np.array([0.0, -0.2], dtype="float32")
        elif action == 2:  # right
            now_action = np.array([0.0, 0.2], dtype="float32")
        # elif action == 3:  # back
        #     now_action = np.array([-0.7, 0.0], dtype='float32')
        else:
            now_action = np.array([0.0, 0.0], dtype="float32")

        self.state, self.reward, _, self.info = self.env.step(now_action)

    def step(self, action):
        # lidar
        if self.state is not None:
            lidar = 5.6 * self.state["scan"].reshape(-1)
            left_dist = np.mean(lidar[-10:])
            middle_dist = np.mean(lidar[330:350])
            right_dist = np.mean(lidar[0:10])
            # print(left_dist, middle_dist, right_dist)

            if not self.turn:
                if left_dist <= 0.2 and middle_dist <= 0.2 and right_dist <= 0.2:  # 0.3
                    self.turn = True
                    if left_dist >= right_dist:
                        self.action = 1
                        self.count = 32
                        # print('left_back')
                    if left_dist < right_dist:
                        self.action = 2
                        self.count = 32
                        # print('right_back')
                    action = self.action

                elif min(left_dist, middle_dist, right_dist) <= 0.25:  # 0.55
                    if middle_dist <= left_dist or middle_dist <= right_dist:
                        self.turn = True
                        if left_dist >= right_dist:
                            self.action = 1
                            self.count = 14
                            # print('turn left')
                        if left_dist < right_dist:
                            self.action = 2
                            self.count = 10
                            # print('turn right')
                        action = self.action
                    if middle_dist > left_dist and middle_dist > right_dist:
                        # print('forward')
                        self.turn = False
                        # self.action = 0
                        # action = self.action
            elif self.turn:
                action = self.action
                self.count -= 1
                if self.count <= 0:
                    self.turn = False

        self.apply_action(action)

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def get_observations(self):
        self.bgr = self.state["rgb"]  # bgr, 144*192*3, 0-1

        return self.bgr

    def get_reward(self, action):
        # live reward
        reward = 0.0
        # collision
        if self.info["collision_step"] > self.collision_step:
            self.collision = True
            reward -= 65
            # self.collision_step = self.info['collision_step']
            # if self.collision_step >= 20:
            #     self.collision = True
            # reward -= 10

        # search reward
        robot_x, robot_y = self.robot.get_position()[:2]
        self.target_distance = np.sqrt(
            (robot_x - self.target_x) ** 2 + (robot_y - self.target_y) ** 2
        )
        if self.target_distance <= self.target_distance_last:
            reward += 0.1
            self.target_distance_last = self.target_distance
        if self.target_distance <= 1:
            reward += 50
            self.done = True

        # lidar
        lidar = 5.6 * self.state["scan"].reshape(-1)
        left_dist = np.mean(lidar[-67:])
        middle_dist = np.mean(lidar[306:374])
        right_dist = np.mean(lidar[0:68])

        min_dist = None
        if action == 0:
            min_dist = (left_dist + middle_dist + right_dist) / 3
            self.min_dist = (self.left_dist + self.middle_dist + self.right_dist) / 3
        if action == 1:
            min_dist = 0.2 * left_dist + 1 / 3 * middle_dist + 7 / 15 * right_dist
            self.min_dist = (
                0.2 * self.left_dist
                + 1 / 3 * self.middle_dist
                + 7 / 15 * self.right_dist
            )
        if action == 2:
            min_dist = 7 / 15 * left_dist + 1 / 3 * middle_dist + 0.2 * right_dist
            self.min_dist = (
                7 / 15 * self.left_dist
                + 1 / 3 * self.middle_dist
                + 0.2 * self.right_dist
            )

        if min(left_dist, middle_dist, right_dist) < 0.7:
            if min_dist - self.min_dist < 0:
                reward += 2 * (min_dist - self.min_dist)
            else:
                reward += self.min_dist - min_dist
        if min(left_dist, middle_dist, right_dist) > 1.5:
            reward += 0.1

        self.left_dist = left_dist
        self.middle_dist = middle_dist
        self.right_dist = right_dist
        self.min_dist = min_dist

        # # forward

        return reward

    def is_done(self):
        return self.done or self.collision
        # return self.collision

    def get_info(self):
        return

    def generate_target_position(self, target_bgr):
        """
        Determine target category from the provided target_bgr and fetch its true location.
        """
        img = np.round(target_bgr * 255).astype(np.uint8)  # 144*192*3, 0-255, bgr
        result = self.yolov3(img).detect_result()
        # categories = ig_categories()
        self.target_category = result[0][0]["label"]

        for i in range(len(self.obj_name_keys)):
            obj_name = self.obj_name_keys[i]
            obj_temp = self.obj_dict[obj_name]
            obj_position = obj_temp.get_position()
            obj_category = obj_temp.category
            if obj_category == self.target_category:
                self.target_x = obj_position[0]
                self.target_y = obj_position[1]
                break

        return

    def reset(self, target_bgr=None):
        # obs
        self.bgr = np.zeros(self.shape)  # 144*192*3; 0-1
        # reward
        self.episodeScore = 0
        self.episodeScoreList = []
        # collision_step
        self.collision_step = 0
        self.collision = False

        # env reset
        self.state = self.env.reset()
        index = np.random.randint(0, len(position_for_reset))
        self.robot.set_position_orientation(
            position_for_reset[index], orientation_for_reset[index]
        )
        self.bgr = self.state["rgb"]

        # target
        # self.target = self.env.task
        self.generate_target_position(target_bgr)
        robot_x, robot_y = self.robot.get_position()[:2]
        self.target_distance = np.sqrt(
            (robot_x - self.target_x) ** 2 + (robot_y - self.target_y) ** 2
        )
        self.target_distance_last = self.target_distance

        # lidar
        self.min_dist = 0
        self.left_dist = 0
        self.middle_dist = 0
        self.right_dist = 0
        #
        self.state = None
        self.reward = None
        self.done = False
        self.info = None
        # Avoid obstacles
        self.action = 0
        self.count = 0
        self.turn = False

        self.current_step = 0

        print("Resetting environment")

        return self.bgr
