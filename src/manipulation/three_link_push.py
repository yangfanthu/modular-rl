import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from utils import *
import os
import numpy as np

class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, xml):
        self.xml = xml
        self.target_pos = np.array([0.4, 0.4,])
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        info = {}
        qpos_before = self.sim.data.qpos
        qvel_before = self.sim.data.qvel
        posbefore = self.sim.data.qpos[0]
        robot_pos_before = self.sim.data.qpos[:-7]
        object_pos_before = self.sim.data.qpos[-7:-4]
        object_orn_before = self.sim.data.qpos[-4:]
        self.do_simulation(a, self.frame_skip)
        object_pos = self.sim.data.qpos[-7:-5]
        distance = np.linalg.norm(np.array(object_pos) - self.target_pos)
        reward = -distance**2
        # torso_height, torso_ang = self.sim.data.qpos[1:3]
        # done = not (torso_height > (0.8 - 0.26) and torso_height < (2.0 - 0.26) and
        #             torso_ang > -1.0 and torso_ang < 1.0)
        distance_to_center = np.linalg.norm(np.array(object_pos[:2]))
        done = False
        if distance_to_center > 2:
            done = True
            reward = -1000
            info = {'suc':False}

        if distance < 0.1:
            done = True
            reward = 1000
            info = {'suc':True}
        ob = self._get_obs()
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
    # def _get_obs(self):
        
    #     def _get_obs_per_limb(b):
    #         if b == 'torso':
    #             limb_type_vec = np.array((1, 0, 0, 0))
    #         elif '1' in b:
    #             limb_type_vec = np.array((0, 1, 0, 0))
    #         elif '2' in b:
    #             limb_type_vec = np.array((0, 0, 1, 0))
    #         elif '3' in b:
    #             limb_type_vec = np.array((0, 0, 0, 1))
    #         else:
    #             limb_type_vec = np.array((0, 0, 0, 0))
    #         torso_x_pos = self.data.get_body_xpos('table')[0]
    #         xpos = self.data.get_body_xpos(b)
    #         xpos[0] -= torso_x_pos
    #         q = self.data.get_body_xquat(b)
    #         expmap = quat2expmap(q)
    #         obs = np.concatenate([xpos, np.clip(self.data.get_body_xvelp(b), -10, 10), \
    #             self.data.get_body_xvelr(b), expmap, limb_type_vec])
    #         # include current joint angle and joint range as input
    #         if b == 'torso':
    #             angle = 0.
    #             joint_range = [0., 0.]
    #         else:

    #             body_id = self.sim.model.body_name2id(b)
    #             jnt_adr = self.sim.model.body_jntadr[body_id]
    #             qpos_adr = self.sim.model.jnt_qposadr[jnt_adr] # assuming each body has only one joint
    #             angle = np.degrees(self.data.qpos[qpos_adr]) # angle of current joint, scalar
    #             joint_range = np.degrees(self.sim.model.jnt_range[jnt_adr]) # range of current joint, (2,)
    #             # normalize
    #             angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    #             joint_range[0] = (180. + joint_range[0]) / 360.
    #             joint_range[1] = (180. + joint_range[1]) / 360.
    #         obs = np.concatenate([obs, [angle], joint_range])
    #         return obs

    #     full_obs = np.concatenate([_get_obs_per_limb(b) for b in self.model.body_names[1:]])
        
    #     return full_obs.ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20