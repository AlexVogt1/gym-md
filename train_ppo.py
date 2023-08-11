from logging import config
import os
import numpy as np                                                   
import gym
from sympy import true   
import gym_md 
import argparse
from stable_baselines3 import PPO  
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
import wandb

def get_reward_scheme(type):
    return


def parse_args():
    parser = argparse.ArgumentParser("PPO for MiniDungeons (gym-md)")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="md-simple_0-v0", help="name of the game")
    # Play-style to train (i.e. which reward scheme to use)
    parser.add_argument("--play_style", type=str, default="optimal", help= "what type of player are you wanting to train")
    parser.add_argument("--reward_scheme", type=str, default="original", help= "what reward scheme are you using for training")
    
    
    # # Core PPO parameters
    # parser.add_argument("--max_ep_len", type=int, default=250, help= "max number of steps allod in single episode")
    # parser.add_argument("--max_training_timesteps", type=int, default=int(3e6), help= "max number of steps for training")
    # parser.add_argument("--k_epochs", type=int, default=80, help= "what k_epochs")
    # parser.add_argument("--eps_clip", type=float, default=0.2, help= "clipping factor")
    # parser.add_argument("--gamma", type=float, default=0.99, help= "gamma value for optimizer")
    # parser.add_argument("--lr_actor", type=float, default=0.0003, help= "learning rate for actor")
    # parser.add_argument("--lr_critic", type=float, default=0.001, help= "learning rate for actor")
    # parser.add_argument("--action_std", type=float, default=0.6, help= "learning rate for actor")
    # parser.add_argument("--action_std_decay_rate", type=float, default=0.05, help= "learning rate for actor")
    # parser.add_argument("--min_action_std", type=float, default=0.1, help= "learning rate for actor")
    # parser.add_argument("--action_std_decay_freq", type=int, default=int(2.5e5), help= "learning rate for actor")
    # #Logging and Reporing
    # parser.add_argument("--print_freq", type=int, default=500, help= "printing frequency")
    # parser.add_argument("--log_freq", type=int, default=500, help= "logging frequency")
    # parser.add_argument("--save_model_freq", type=int, default=int(1e5), help= "frequency for saving model")
    return parser.parse_args()

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# def parse_args():
#     parser = argparse.ArgumentParser("PPO for MiniDungeons (gym-md)")
#     parser.add_argument("--seed", type=int, default=0, help="which seed to use")
#     # Environment
#     parser.add_argument("--env", type=str, default="md-simple_0-v0", help="name of the game")
#     # Play-style to train (i.e. which reward scheme to use)
#     parser.add_argument("--play_style", type=str, default="optimal", help= "what type of player are you wanting to train")
#     parser.add_argument("--reward_scheme", type=str, default="original", help= "what reward scheme are you using for training")
    
    
#     # # Core PPO parameters
#     # parser.add_argument("--max_ep_len", type=int, default=250, help= "max number of steps allod in single episode")
#     # parser.add_argument("--max_training_timesteps", type=int, default=int(3e6), help= "max number of steps for training")
#     # parser.add_argument("--k_epochs", type=int, default=80, help= "what k_epochs")
#     # parser.add_argument("--eps_clip", type=float, default=0.2, help= "clipping factor")
#     # parser.add_argument("--gamma", type=float, default=0.99, help= "gamma value for optimizer")
#     # parser.add_argument("--lr_actor", type=float, default=0.0003, help= "learning rate for actor")
#     # parser.add_argument("--lr_critic", type=float, default=0.001, help= "learning rate for actor")
#     # parser.add_argument("--action_std", type=float, default=0.6, help= "learning rate for actor")
#     # parser.add_argument("--action_std_decay_rate", type=float, default=0.05, help= "learning rate for actor")
#     # parser.add_argument("--min_action_std", type=float, default=0.1, help= "learning rate for actor")
#     # parser.add_argument("--action_std_decay_freq", type=int, default=int(2.5e5), help= "learning rate for actor")
#     # #Logging and Reporing
#     # parser.add_argument("--print_freq", type=int, default=500, help= "printing frequency")
#     # parser.add_argument("--log_freq", type=int, default=500, help= "logging frequency")
#     # parser.add_argument("--save_model_freq", type=int, default=int(1e5), help= "frequency for saving model")
#     return parser.parse_args()

# def set_rewards()

def main(lvl, exp, steps, log_dir):
# def main(lvl,, log_dir):

    env = gym.make(f"md-{lvl}-v0")       
    # env = Monitor(env, log_dir)                                                    
                                                                                            
    model = PPO(policy = "MlpPolicy",env =  env, verbose=1, device="cuda", tensorboard_log=log_dir)   
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)                        
    model.learn(total_timesteps=steps,callback=callback)                                                      
                                                                                            
    # model.save("ppo_cartpole")  # saving the model to ppo_cartpole.zip                      
    # model = PPO.load("ppo_cartpole")  # loading the model from ppo_cartpole.zip             
                                                                                            
    # obs = env.reset()                                                                       
    # for i in range(1000):                                                                   
    #     action, _state = model.predict(obs, deterministic=False)                             
    #     obs, reward, done, info = env.step(action) 
    #     print(info)                                         
    #     # env.render(mode='human')                                                                        
    #     if done:                                                                            
    #         obs = env.reset()


if __name__ == '__main__':
    args = parse_args()
    # config = {
    #     "env": args.env,
    #     "play_style": args.play_style,
    #     "reward_sheme": args.reward_scheme,
    #     "experiment": "test",
    #     # "seed":args.seed,
    #     # "max_ep_len": args.max_ep_len,
    #     # "max_training_timesteps":args.max_training_timesteps,
    #     # "k_epochs":args.k_epochs,
    #     # "eps_clip":args.eps_clip,
    #     # "gamma": args.gamma,
    #     # "lr_actor":args.lr_actor,
    #     # "lr_critic": args.lr_critic,
    #     # "action_std":args.action_std,
    #     # "action_std_decay_rate":args.action_std_decay_rate,
    #     # "min_action_std":args.min_action_std,
    #     # "action_std_decay_freq":args.action_std_decay_freq,
    #     # "print_freq":args.print_freq,
    #     # "log_freq": args.log_freq,
    #     # "save_model_freq": args.save_model_freq
    # }
    print(args)

    config ={
        "lvl": args.env,
        "play_style": args.play_style,
        "reward_sheme": "original",
        "experiment": "test",
    }
    # return
    log_dir = "TESTING_DUMP/"
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(project="gym-md_SB3_test", sync_tensorboard=True)
    main(lvl= "simple_0", log_dir=log_dir)
    

