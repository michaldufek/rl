import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
import yfinance as yf
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from ray.rllib.models.preprocessors import get_preprocessor

class DataSimulator():
    '''
    Simulate fake data for sanity check of deep reinforcement learning algorithms and environments.
    
    Parameters
    ----------
    SO : float, default=100
        Initial stock price
    mu : float, defaul=0.02
        Average return
    sigma : float, default=0.05
        Standatd deviation (volatility)
    N : int, default=5
        Number of stocks/simulated trajectiories
    days : int, default=252
        Length of simulated trajectiories
    plot : bool, default=True 
        Plot simulated data.
    name_prefix: str, defalult='fake'
        Name prefix for specific simulated series.

    ### Data Simulating Module for Sanity Check ###
     -- dummy data for sanity check of deep reinforcement learning algorithms and environments <br>
         -- based on input parameters, create "good" and "bad" data <br>
     -- simulated data based on calibrated distribution function to enrich datasets <br>
         -- 
    ##### Dummy data simulation ###
    Stochastic differential equation for the process is: <br>

    $\frac{dP_{t}}{P_{t}} = µdt + σdW_{t}$ <br>
    Closed-form integration of the SDE: <br>
    St = $S_{0}exp(\int_{0}^{t}[\mu - \frac{1}{2}\sigma^2]ds + \int_{0}^{t}\sigma dW_{s})$
    '''
    def __init__(self, S0=100., mu=0.02, sigma=0.05, N=5, days=252*1, plot=True, name_prefix='fake'):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.days = days
        self.T = 1
        self.dt = self.T / self.days
        self.plot = plot
        self.name_prefix = name_prefix

        self.S, self.names = self.simulate_dummies()

    def simulate_dummies(self):
        S = np.zeros((self.days+1, self.N))
        S[0] = self.S0

        for t in range(1, self.days+1):
            W = np.random.normal(size=self.N) # random part of the stochastic equation (wiener process)
            S[t] = S[t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * W) # St = S0exp()

        if self.plot:
            plt.figure(figsize=(15, 10))
            plt.plot(S)
            plt.show()

        captions = [ f'{self.name_prefix}_{i+1}' for i in range(self.N)]

        return S, captions


class StockPickerEnv(gym.Env):
    def __init__(self,
                #tickers=['ES=F', 'IBM', 'VOW.DE', 'VRTX'],
                tickers = ['F', 'VRTX', 'AMZN'],
                initial_amount=100000, 
                mkt_position_thresholds=(0,2), 
                leverage_threshold=2,
                N=3,
                max_stocks=4):
        # Atributes
        self.tickers = tickers
        self.N = N # dummy stocks 'good' and 'bad'; radek is real badass
        self.df = self._get_data()
        self.initial_amount = initial_amount
        self.n_stocks = 0
        self.index = 0
        self.transactions_cost_pct = 0.0015
        self.portfolio_value = self.initial_amount
        self.done = False
        self.features_list = ['LogReturns']
        self.mkt_position_thresholds = mkt_position_thresholds
        self.leverage_threshold = leverage_threshold
        self.total_market_position = 0
        self.leverage = 1
        self.max_stocks = max_stocks
        self.action_number = len(self.tickers) + 2 * self.N if self.N > 0 else len(self.tickers) # N is for dummy stocks: 2* because stocks are 'good' and 'bad'
        
        # Acton-State space
        self.action_space = spaces.Dict({
                            "total_market_position": spaces.Box(low=0, high=2, shape=()),
                            "leverage": spaces.Box(low=1, high=2, shape=()),
                            "orders": spaces.MultiDiscrete([3]*9)
                            #"orders": spaces.MultiDiscrete([3]*self.action_number)
                            })
        '''
        self.observation_space = spaces.Dict({
                                #"features": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.Symbol.unique()), )),  #len(self.df.columns)-2
                                "features": spaces.Box(low=0, high=1, shape=(8,)),
                                "n_stocks":spaces.Discrete(15)
                        })
        '''
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # Memory Buffer
        self.reward_memory = [0] 
        self.actions_memory = len(self.df.Symbol.unique()) * [0] # no actions at the beginning
        self.date_memory =  [self.df.Date.unique()[0]] # first date where the action and reward is nothing       

    def step(self, action):
        print(f'A C T I O N {action["orders"]}')
        n_longs = len([ x for x in action['orders'] if x > 0 ])
        n_shorts = len([ x for x in action['orders'] if x < 0 ])
        self.n_stocks = len([ x for x in action['orders'] if x != 0 ]) # beacause of gym.spaces.Discrete()
        
        self.total_market_position = action['total_market_position']
        self.leverage = action['leverage']
        
        # Load next state (data for next day)
        self.index += 1
        previous_day_data = self.df[self.df.Date==self.df.Date.unique()[self.index-1]]
        current_day_data = self.df[self.df.Date==self.df.Date.unique()[self.index]]
        features_data = current_day_data[self.features_list].values.flatten()
        
        '''
        self.state = { "features": features_data,
                        "n_stocks": self.n_stocks
                    }
        self.state = self.flatten_obs()
        '''
        self.state = np.float32(features_data)
        #print(f'state from step: {self.state}')

        # incoming parameter "action" brings i.a. list of orders (as actuall picks)
        picks_ = []
        actionMapper = {
            0:0,
            1:1,
            2:-1
        }
        [ picks_.append(actionMapper[pick]) for pick in action['orders'] ]

        ### Capital allocation ###
        # Array with allocations -- portfolio items values at the start of the trading period
        long_weight, short_weight = self._get_weights()
        nav_per_long = long_weight * self.portfolio_value
        nav_per_short = short_weight * self.portfolio_value
        allocations = np.array(picks_)
        allocations = np.where(allocations > 0, allocations*nav_per_long, allocations)
        allocations = np.where(allocations < 0, allocations*nav_per_short, allocations)
        # Calculation of results for specific time period -- return rate, profit, new portfolio values
        change_rate = current_day_data.Close.values / previous_day_data.Close.values
        _rewards = change_rate * allocations - allocations # aka profit for each asset
        reward = np.sum(_rewards)
        
        self.reward_memory.append(reward)
        #profit_rate = reward / self.portfolio_value
        self.portfolio_value += reward

        self.done = bool(self.index >= len(self.df.Date.unique()) - 2) # explanation: e.g. length is 100 => 100 - 1 to receive correct index value and 99 - 1 to switch done = True before the last step (else in the next step the index is out of range). In other words, action is taken at the penultimate step, ultimate step i terminal
        
        if self.done:
            daily_returns = pd.DataFrame(self.reward_memory)
            daily_returns.columns = ["Daily_return"]
            #plt.plot(daily_returns.Daily_return.cumsum(), 'r')
            #plt.savefig('results*cumulative.png')
            #plt.close()

            #plt.plot(self.reward_memory, 'r')
            #plt.savefig('results/portfolio_value.png')
            #plt.close()

            print("====================================")
            print(f"end_total_asset: {self.portfolio_value}")

            #print(f'reward memory: {self.reward_memory}')
            if daily_returns['Daily_return'].std() != 0:
                sharpe = (252**0.5) * daily_returns['Daily_return'].mean() / daily_returns['Daily_return'].std()
                print(f'sharpe: {sharpe}')

        return self.state, self.reward_memory[-1], self.done, {'Terminal step': self.done}
    
    def _get_data(self, start='2010-01-01'):
        '''
        Price data download. 
        If self.sanity_check is True, the method simulates "good" and "bad" stocks and add them to the outcoming dataframe.
        Simulated data serves as a sanity check for data mining and optimization algorithms (one can intuitively see whether the algos work).
        '''    
        if self.tickers != []:

            #df = yf.download(self.tickers, start=start)
            df = yf.download(['F', 'VRTX', 'AMZN'])
            df = df.Close
            df = df.dropna().reset_index()
            df['Date'] = df['Date'].astype(str)
            #df = df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')

        if self.N > 0:
            good_data = DataSimulator(mu=0.18, sigma=0.04, N=self.N, days=len(df.Date.unique())-1, name_prefix='good_stock', plot=False)
            good_df = pd.DataFrame(data=good_data.S, columns=good_data.names)
            good_df['Date'] = df.Date.unique()
            #good_df = good_df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
            #print(f'Good data: {good_df}')
            bad_data = DataSimulator(mu=-0.02, sigma=0.35, N=self.N, days=len(df.Date.unique())-1, name_prefix='bad_stock', plot=False)
            bad_df = pd.DataFrame(data=bad_data.S, columns=bad_data.names)
            bad_df['Date'] = df.Date.unique()
            #bad_df = bad_df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
            #print(f'Bad data: {bad_df}')
            df = pd.concat(objs=[df, good_df, bad_df], axis='columns')
            df = df.loc[:, ~df.columns.duplicated()].copy() # Date column is duplicated
            df = df.reset_index(drop=True) # duplicity in index

        closePrices = df.drop(['Date'], axis=1) #keep only numbers for returns calculation
        logReturns = np.log(closePrices / closePrices.shift(1))

        scaler = MinMaxScaler()
        scaler.fit(logReturns.T)
        scaledReturns = scaler.transform(logReturns.T).T # cross-sectional scaling
        #print(logReturns.columns)
        scaledReturns = pd.DataFrame(scaledReturns, columns=logReturns.columns)
        scaledReturns['Date'] = df.Date.unique()

        df = df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
        scaledReturns = scaledReturns.melt(id_vars=['Date'], var_name='Symbol', value_name='LogReturns')

        df = pd.concat(objs=[df, scaledReturns], axis='columns')
        df = df.loc[:, ~df.columns.duplicated()].copy() # drop duplicate columns
        print(df[df.Date==df.Date.unique()[-1]])        

        return df
    
    def _get_weights(self):
        '''
        The method returns weight number for both long and short position. 
        There is an assumption that all longs (shorts) are equally weighted (warning: simultaneously longs != shorts).
        
        Relationship between 'leverage' and 'market exposition' :
        ---------------------------------------------------------
        leverage = |longs| + |shorts|
        market_exposition = longs + shorts

        One can infer 'long' and 'short' position to keep both requirements.
        '''
        if self.total_market_position < self.mkt_position_thresholds[0] or self.total_market_position > self.mkt_position_thresholds[1]:
            raise ValueError(f"Market position exceeds thresholds: {self.total_market_position} is out of the bounds {self.mkt_position_thresholds}")
        elif self.leverage < 1 or self.leverage > self.leverage_threshold:
            raise ValueError(f"Financial leverage: {self.leverage} is out of the bounds: 1, {self.leverage_threshold}")
        
        long = (self.leverage + self.total_market_position) / 2
        short = self.total_market_position - long
        long_weight = long / (self.n_stocks/2) # long share divided by number of stocks on long side (long/(n_stock/2)) 
        short_weight = short / (self.n_stocks/2) # the same but for the short side

        return long_weight, short_weight               

    def reset(self):
        #print(f'sanity check mode {self.N}')

        self.index = 1
        #print(f'index from reset {self.index}')
        self.done = False
        self.n_stocks = 0
        self.portfolio_value = self.initial_amount
        self.total_market_position = 0
        self.leverage = 0

        self.reward_memory = [0]
        self.actions_memory = len(self.df.Symbol.unique()) * [0]
        self.date_memory = [self.df.Date.unique()[0]]

        # Initial state
        features_data = self.df[self.df.Date==self.df.Date.unique()[self.index]][self.features_list].values.flatten()
        #print(features_data.shape)
        '''
        self.state = { "features": features_data,
                        "n_stocks": self.n_stocks
        }

        self.state = self._flatten_obs()
        '''
        self.state = np.float32(features_data)

        #self.state = self.flatten_obs()
        #print(f'state from reset: {self.state.shape}')
        #print(f'obs space from reset {self.observation_space}')
        #print(f'dtype from reset: {self.state.dtype}')
        return self.state

    def flatten_obs(self):
        return (get_preprocessor(self.observation_space)(self.observation_space)).transform(self.state)


# EoF