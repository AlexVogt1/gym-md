
"""agent module."""
from email import policy
from typing import List, Tuple,Final
import random
from random import Random
import os

from gym_md.envs.agent.actioner import Actioner, Actions
from gym_md.envs.agent.move_info import MoveInfo
from gym_md.envs.agent.pather import Pather
from gym_md.envs.grid import Grid
from gym_md.envs.point import Point
from gym_md.envs.setting import Setting
from gym_md.envs.agent.agent import Agent


class PolicyAgent(Agent):
    """Agent class.

    エージェントクラス．
    Pather, Actionerを持つ．
    """

    def __init__(self, grid: Grid, setting: Setting, random: Random, action_type: str):
        super().__init__(grid, setting, random)
        self.action_type = action_type

    # def reset(self) -> None:
    #     """エージェントの初期化をする.
    #     Reset/Initialize the agent

    #     Notes
    #     -----
    #     リセット形式にしている．
    #     これは，一回一回Agentインスタンスを作ると，rendererなどの参照とずれてしまうため
    #     - in reset format
    #     - This is because if you create an Agent instance once, it will be misaligned with references such as renderer

    #     """
    #     self.hp = self.setting.PLAYER_MAX_HP
    #     init_pos: Final[Point] = self._init_player_pos()
    #     self.y: int = init_pos[0]
    #     self.x: int = init_pos[1]
    def get_policy_action(self, policy: str) -> str:
        # take policy find path for policy
        path = ""
        # load policy
        # predict

        return 'action'

    def select_action(self, actions: Actions) -> str:
        if self.action_type == 'policy':
            return self.select_policy_action(actions)
        else:
            return super().take_action(actions)

    def select_policy_action(self, actions: Actions) -> str:
        # takes in a list of real values the size of the number of polices
        actions_idx: List[Tuple[float, int]] = [(actions[i], i) for i in range(len(actions))]
        actions_idx.sort(key=lambda z: (-z[0], -z[1]))

        max_value = max(actions)
        max_actions = [i[1] for i in actions_idx if i[0]==max_value]
        random.shuffle(max_actions)

        # NUM_TO_POLICY_ACTION =  {0: 'TREASURE', 1: 'KILLER', 2: 'RUNNER', 3: 'POTION'}
        action_out = self.setting.NUM_TO_POLICY_ACTION[max_actions[0]]
        return action_out

        return
    def take_action(self, action: str) -> None:
        if self.action_type == 'policy':
            return self.take_policy_action(action)
        else:
            return super().take_action(action)
        
    def take_policy_action(self, policy:str):
        if policy == 'TREASURE':
            action = get_policy_action(policy)
            super().take_action(action)
        elif policy == 'KILLER':
            action = get_policy_action(policy)
            super().take_action(action)
        elif policy == 'RUNNER':
            action = get_policy_action(policy)
            super().take_action(action)
        elif policy == 'POTION':
            action = get_policy_action(policy)
            super().take_action(action)
        else:
            print("ERROR: POLICY DOES NOT EXIST -> policy_agent -> take_policy_action")
            


    def is_dead(self) -> bool:
        return self.hp <= 0
    

    # def is_exited(self) -> bool:
    #     return self.grid[self.y, self.x] == self.setting.CHARACTER_TO_NUM["E"]

    # def be_influenced(self, y: int, x: int) -> None:
    #     """移動したプレイヤーに影響を与える.
    #     Affect moved players

    #     体力の増減を行う
    #     Increase/decrease health

    #     Parameters
    #     ----------
    #     y:int
    #         移動後のy
    #     x:int
    #         移動後のx
    #     """
    #     if self.grid[y, x] == self.setting.CHARACTER_TO_NUM["M"]:
    #         attack = self.setting.ENEMY_POWER
    #         if self.setting.IS_ENEMY_POWER_RANDOM:
    #             attack = self.random.randint(
    #                 self.setting.ENEMY_POWER_MIN, self.setting.ENEMY_POWER_MAX
    #             )
    #         self.hp -= attack
    #     if self.grid[y, x] == self.setting.CHARACTER_TO_NUM["P"]:
    #         self.hp += self.setting.POTION_POWER
    #         if self.setting.IS_PLAYER_HP_LIMIT:
    #             self.hp = min(self.hp, self.setting.PLAYER_MAX_HP)

    # def change_player_hp(self, previous_hp: int) -> None:
    #     """前回のステージのHPに更新する。"""
    #     self.hp = min(previous_hp, self.setting.PLAYER_MAX_HP)

    # def _init_player_pos(self) -> Point:
    #     """プレイヤーの座標を初期化して座標を返す.
    #     Initialize the player's coordinates and return the coordinates.
        
    #     Notes
    #     -----
    #     初期座標を表すSを'.'にメソッド内で書き換えていることに注意する．
    #     Note that S representing the initial coordinates is rewritten to '.' in the method

    #     Returns
    #     -------
    #     Point
    #         初期座標を返す
    #         return the initial coordinates
    #     """
    #     for i in range(self.grid.H):
    #         for j in range(self.grid.W):
    #             if self.grid[i, j] == self.setting.CHARACTER_TO_NUM["S"]:
    #                 self.grid[i, j] = self.setting.CHARACTER_TO_NUM["."]
    #                 return i, j
