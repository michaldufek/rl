from enum import unique
import os
from symtable import Symbol
from stockpickerenv import DataSimulator, StockPickerEnv
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

import matplotlib.pyplot as plt

config = {
    'env': StockPickerEnv, # not registred so use class name
    'env_config': None,
    'num_gpus': int(os.environ.get("RLLIB_NUMGPUS", "0")),
    'model': {'max_seq_len': 128},
    'num_workers': 1, # no parallelism
    'framework': 'tf2',
}
ppo_config = DEFAULT_CONFIG.copy()
ppo_config.update(config)
ppo_config['num_workers'] = 0
ppo_config['train_batch_size'] = 128
ppo_config['num_sgd_iter'] = 10
ppo_config['framework'] = 'tf2'
ppo_config['eager_tracing'] = True
ppo_config['soft_horizon'] = True
ppo_config['disable_env_checking'] = True
ppo_config['_disable_preprocessor_api'] = True
ppo_config['horizon'] = 2000
ppo_config['ignore_worker_failures'] = True
ppo_config['vf_clip_param'] = 100000

iterations = 30
training = True

def check_denied_action(action):
    '''Return True for denied and undesired actions'''
    return bool(
        len([ x for x in action['orders'] if x != 0 ]) == 0 or # at minimum one long and one short
        len([ x for x in action['orders'] if x == 1 ]) != len([ x for x in action['orders'] if x == 2 ]) or # number of longs must be same as shorts
        len([ x for x in action['orders'] if x != 0 ]) > env.max_stocks # long and short positions are not allowed to exceed maximum number of stocks
    )

if __name__=='__main__':
    env = StockPickerEnv()
    #print(f'Action space: {env.action_space}')
    agent = PPOTrainer(config=ppo_config, env=StockPickerEnv)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    step = 0
    iteration = 0

    if training:
        for iteration in range(iterations):
            result = agent.train()
            print(result['episode_reward_mean'])

    else:
        for iteration in range(iterations):
            done = False
            print(' -- main')
            obs = env.reset()
            cumulative_reward = 0
            steps, cumulative_rewards = [], []
            while not done:
                step += 1
                
                if training:
                    print(' -- TRAINING STARTS')
                    results = agent.train()
                    print(f'Iter: {step}, reward: {results["episode_reward_mean"]}')
                    print(' -- TRAINING SUCCESFULLY DONE')

                else:
                    action = agent.compute_single_action(obs)
                    while check_denied_action(action=action):
                        action = agent.compute_single_action(obs)
                        #check_denied_action(action)
                        
                    obs, reward, done, info = env.step(action)
                    
                    ax.plot(steps, cumulative_rewards, color='b')
                    fig.canvas.draw()
                    fig.show()

                    cumulative_reward += reward
                    cumulative_rewards.append(cumulative_reward)
                    steps.append(step)

                    if step % 100 == 0:
                        print(f'Step: {step}')
                        print(f'...choosen action: {action["orders"]} for stocks: {env.df.Symbol.unique().tolist()}')
                        print(f'received reward: {reward}')

                        print('*****************************************')
                        print(f'ITERATION NUMBER: {step}')
                        print(f'CUMULATIVE REWARD: {cumulative_reward}')
                        print('*****************************************')

                    plt.pause(0.01)


