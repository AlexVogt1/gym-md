from random import Random
from typing import DefaultDict, Final, List, Tuple, Dict
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
from gym_md.envs.grid import Grid
from gym_md.envs.point import Point
from gym_md.envs.setting import Setting
from gym_md.envs.definition import POLICY_ACTIONS
from gym_md.envs.grid import Grid

# from gym_md.envs.renderer.collab_renderer import CollabRenderer

#TODO change Action Space
#TODO Create function to cread in policies from model.zip and use is as actions
#TODO create function that creates the policy


class MdPolicyEnv(MdEnvBase):
    def __init__(self, stage_name: str, action_type='path', policy_path = "../play_style_models/base/"):

        self.random = Random()
        # self.stage_name: Final[str] = stage_name

        # self.setting: Final[Setting] = Setting(self.stage_name)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)

        self.grid: Grid = Grid(self.stage_name, self.setting)
        self.info: DefaultDict[str, int] = defaultdict(int)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.setting.DISTANCE_INF, shape=(16,), dtype=numpy.int32
        )
        self.action_type = action_type
        if action_type=='switching':
            self.agent: PolicyAgent =PolicyAgent(self.grid, self.setting, self.random, self.action_type)
            #TODO make action space be length of policy_names wrapped i.e. length of policy actions
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)
            try: 
                self.setting.POLICY_ACTIONS = self.agent.actions
                self.setting.POLICY_ACTION_TO_NUM = Setting.list_to_dict(self.setting.POLICY_ACTIONS)
                self.setting.NUM_TO_POLICY_ACTION = Setting.swap_dict(self.setting.POLICY_ACTION_TO_NUM)
            except:
                self.setting.POLICY_ACTIONS = POLICY_ACTIONS
                self.setting.POLICY_ACTION_TO_NUM = Setting.list_to_dict(self.setting.POLICY_ACTIONS)
                self.setting.NUM_TO_POLICY_ACTION = Setting.swap_dict(self.setting.POLICY_ACTION_TO_NUM)
                # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)                
        else:
            self.agent: Agent = Agent(self.grid, self.setting, self.random)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))

    def reset(self) -> List[int]:
        super().reset()
        self.agent.reset()
        return
    
    def step(self, actions):

        return observation, reward, done, self.info

    
    
