from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state2 import ObservableState

import numpy as np

class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent
        self.id = None # the human's ID, used for collecting traj data
        self.observed_id = -1 # if it is observed, give it a tracking ID

        self.detector_px = None
        self.detector_py = None
        self.detector_vx = None
        self.detector_vy = None

    # ob: a list of observable states
    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    # ob: a joint state (ego agent's full state + other agents' observable states)
    def act_joint_state(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = self.policy.predict(ob)
        return action

    def get_detected_state(self):
        # return ObservableState(-6, -6, -6, -6, 0.2)
        return ObservableState(self.detector_px, self.detector_py, self.detector_vx, self.detector_vy, self.radius)

    def set_detected_state(self, position, time_step):
        if self.detector_px is None or self.detector_py is None:
            raise AttributeError("Need to set detector px and py before calling set_detected_state")
        
        self.set_detected_position(position)
        old_pos = np.array([self.detector_px, self.detector_py])
        new_vel = (position - old_pos) / time_step
        self.set_detected_velocity(new_vel)

    def set_detected_position(self, position):
        self.detector_px = position[0]
        self.detector_py = position[1]

    def set_detected_velocity(self, velocity):
        self.detector_vx = velocity[0]
        self.detector_vy = velocity[1]
