from random import Random
from typing import DefaultDict, Final, List, Tuple
from PIL import Image
import numpy
import gym
from collections import defaultdict


from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.policy_agent import PolicyAgent
from gym_md.envs.agent.agent import Agent
# from gym_md.envs.agent.companion_agent import CompanionAgent, DirectionalAgent
# from gym_md.envs.renderer.collab_renderer import CollabRenderer
from gym_md.envs.agent.actioner import Actions
from gym_md.envs.point import Point
from gym_md.envs.setting import Setting
from gym_md.envs.definition import POLICY_ACTIONS
from gym_md.envs.grid import Grid
# from gym_md.envs.renderer.collab_renderer import CollabRenderer

#TODO change Action Space
#TODO Create function to cread in policies from model.zip and use is as actions
#TODO create function that creates the policy


class MdPolicyEnv(MdEnvBase):
    def __init__(self, stage_name: str, action_type='path'):

        #Action Space
        # self.setting: Final[Setting] = Setting(self.stage_name)
        # self.setting.POLICY_ACTIONS: Final[List[str]] = POLICY_ACTIONS
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)
    


    
    
