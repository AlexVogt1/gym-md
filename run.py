import gym
import gym_md
import random
import matplotlib.pyplot as plt
# from gym_md.envs.agent.agent import Agent


def save_visitaion(x,y,lvl,play_style,ep):
    coords =[]
    for i,j in zip(x,y):
        coords.append([i,j])

    file = open(F'/home/alex/test/data/{lvl}-{play_style}-{ep}.txt', 'w')
    for items in coords:
        line = ','.join(str(x) for x in items)
        file.write(line + '\n')
    
    return coords

play_styles = {'exit_safely_actions':       [0, 4, 2, 3, -5, 1, 8],
               'monster_actions':           [-1, -2, -2, -2, -2, -2, -2],
               'treasure_safely_actions':   [0, 0, 2, 0, 0, 0, 1],
               'potion_safely_actions':     [0, 0, 0, 0, 4, 0, 1]
               }

# make env using level
play_style = 'potion_safely_actions'
lvl = 'holmgard_7'
env = gym.make(f'md-{lvl}-v0')


x=[]
y=[]
LOOP: int = 150
TRY_OUT: int = 30
frame =[]
for _ in range(TRY_OUT):
    observation = env.reset()
    reward_sum = 0
    frame =[]
    for i in range(LOOP):
        x.append(env.agent.x)
        y.append(env.agent.y)
        
        # env.render(mode='human')
        # frame.append(env.render(mode='rgb_array'))
        # actions = [random.random() for _ in range(7)]
        actions = play_styles[play_style]
        observation, reward, done, info = env.step(actions)
        reward_sum += reward

        if done:
            env.render()
            break

    print(reward_sum)
    print(observation)
    print(info)
print(save_visitaion(x,y,lvl,play_style,TRY_OUT))