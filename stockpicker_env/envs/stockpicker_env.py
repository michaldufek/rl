import ray, gym
from gym import spaces

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class StockPickerEnv(gym.Env):
    def __init__(self,
                df,
                stock_dim,
                initial_amount,
                tranaction_cost_pct,
                gamma,
                features_list,
                #state_space,
                market_pos_threshold,
                leverage_threshold,
                max_stocks
                ):
        
        self.df = df
        self.day = 0
        self.time_dim = self.df.date.unique()
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.tranaction_cost_pct = tranaction_cost_pct
        self.gamma = gamma
        self.features_list = features_list
        self.feature_dim = len(features_list)
        self.market_pos_threshold = market_pos_threshold
        self.leverage_threshold = leverage_threshold
        self.max_stocks = max_stocks

        #self.state_space = state_space # ?
        self.action_space = spaces.Dict({
                        "total_market_position": spaces.Box(low=0, high=2, shape=(1,)), #Buy/Sell/Do nothing for each stock in universe
                        "leverage": spaces.Box(low=1, high=2, shape=(1,)),
                        "picks": spaces.MultiDiscrete([3]*self.stock_dim)}) # 3: 0 Do Nothing, 1 Buy, 2 Sell
        self.observation_space = spaces.Dict({
                                    'covariation_matrix': spaces.Box(low=-np.inf, high=np.inf, shape=(self.stock_dim, self.stock_dim)),
                                    'features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.stock_dim, self.feature_dim)),
                                    # might be part of action
                                    'total_market_position': spaces.Box(low=0, high=2, shape=(1,)),
                                    'leverage': spaces.Box(low=1, high=2, shape=(1,))
                                    })
        # data for each observation
        self.data = self.df[self.df.date == self.time_dim[self.day]]
        self.covs = self.data['cov_list'].values[0]
        self.total_market_position = 0
        self.leverage = 1
        self.n_stocks = 5 # minimum stocks in portfolio: 5, max is self.max_stocks

        self.state_space = spaces.Dict({
                            'covariation_matrix': spaces.Box(low=-np.inf, high=np.inf, shape=(self.stock_dim, self.stock_dim)),
                            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.stock_dim, self.feature_dim)),
                            # might be part of action
                            'total_market_position': spaces.Box(low=0, high=2, shape=(1,)),
                            'leverage': spaces.Box(low=1, high=2, shape=(1,))
                            })

        self.terminal = False
        self.portfolio_value = self.initial_amount

        # memory for each step
        self.asset_memory = [ self.initial_amount ]
        self.portfolio_return_memory = [0]
        self.actions_memory = [ [0]*self.stock_dim ] # initial action is DO NOTHING
        self.date_memory = [ self.data.date.unique()[0] ]
        self.market_pos_memory = self.total_market_position
        self.leverage_memory = [ self.leverage ]
        self.n_stocks_memory = [ self.n_stocks ]

    def step(self, actions):
        n_longs = len([ x for x in actions['picks'] if x > 0 ])
        n_shorts = len([ x for x in actions['picks'] if x < 0 ])
        self.n_stocks = len([ x for x in actions['picks'] if x != 0 ])

        # add constraints to your problem (i.e. n_shorts == n_longs, n_longs < max_longs, total_market_exposion < max_exposion, leverage <= max_leverage, etc.)
        self.terminal = bool(
            self.day >= len(self.time_dim) - 1
            or n_shorts != n_longs # number of shrots has to be equal to number of longs
            or n_shorts < 1 # 1 stock in portfolio at minimum
            or n_shorts < self.max_stocks # number of stocks in portfolio should not exceed specific number
            or self.leverage < self.total_market_position) # short positions must be always negative (otherwise you will buy it)
        
        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results*cumulative_reward.png')
            plt.close()

            print('=======================================')
            print(f'begin_total_asset: {self.asset_memory[0]}')
            print(f'end_total_asset: {self.portfolio_value}')

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252**0.5)*df_daily_return['daily_return'].mean() / df_daily_return['daily_return'].std()
                print('sharpe: ', sharpe)
            print('=======================================')

            return self.state, self.reward, self.terminal, {}

        else:
            last_day_memory = self.data

            # load next state (data for next day)
            self.day += 1
            self.data = self.df[self.df.date == self.time_dim[self.day]]
            self.covs = self.data['cov_list'].values[0]

            self.n_stocks = len([ x for x in actions['orders'] if x != 0 ])
            self.total_market_position = actions['total_market_position'][0]
            self.leverage = actions['leverage'][0]

            #self.state = np.append(np.array(self.covs), [ self.data[feature] for feature in self.features_list ], axis=0)        
            self.state = {'covariation_matrix': self.covs,
                                'features': self.data[self.features_list].values,
                                'total_market_position': np.array(self.total_market_position),
                                'leverage': np.array(self.leverage),
                                'n_stocks': np.array(self.n_stocks)
                            }
            
            actions_BSN = [] # object for intuitive understanding of actions: 0: do nothing, 1: buy, -1: sell 
            actionMapper = {
                0:0,
                1:1,
                2:-1
            }
            [ actions_BSN.append(actionMapper[action]) for action in actions['orders'] ]
            # portfolio_return = close_price - previous_close_price
            detail_returns = (actions_BSN * self.data.close.values) / (actions_BSN * last_day_memory.close.values)
            portfolio_return = detail_returns[~np.isnan(detail_returns)].mean() # only bought/sell (actions: 1, 1) stock taken into account
            new_portfolio_value = self.portfolio_value * portfolio_return
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.actions_memory.append(actions_BSN) # intuitive representation of actions
            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, {}

    ### RESET ###
    def reset(self):
        self.asset_memory = [ self.initial_amount ]
        self.day = 0
        self.data = self.df[self.df.date == self.time_dim[self.day]]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.total_market_position = 0
        self.leverage = 1
        self.n_stocks = 0
        self.state = {'covariation_matrix': self.covs,
                        'features': self.data[self.features_list].values,
                        'total_market_position': np.array(self.total_market_position),
                        'leverage': np.array(self.leverage),
                        'n_stocks': np.array(self.n_stocks)
                    }
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [ [0]*self.stock_dim ] # DO NOTHING initialy
        self.date_memory = [ self.data.date.unique()[0] ]

        return self.state

    ### RENDER ### 
    def render(self, mode='human'):
        return self.state

    ### SAVE ASSET MEMORY ###
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list, 'daily_return':portfolio_return})

        return df_account_value

    ### SAVE ACTION MEMORY ###
    def save_action_memory(self):

        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date

        return df_actions

    ### SEED ###
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [ seed ]

    ### GET STABLE BASELINE ENVIRONMENT ###
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
# EoF