from typing import Any
import numpy as np
import numpy.typing as npt
from tabox.momentum import price_percentile
from tabox.limit import crop
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from mlbox.agent.a2c.discrete import A2CDiscreteAgent
from mlbox.agent.a2c.nn.lstm import LSTM_ActorCriticDiscrete
from mlbox.trenv import BasicTrEnv
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.strategy.types import Hook
from trbox.trader import Trader
from typing_extensions import override

# train
TRAIN_START = '2017-01-01'
TRAIN_END = '2019-12-31'
# validation
VALD_START = '2020-01-01'
VALD_END = '2021-12-31'
# test
TEST_START = '2020-01-01'
TEST_END = '2023-03-31'

# params
SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
LENGTH = 200
INTERVAL = 5
START_LV = 0.01
N_FEATURE = 50
FREQ = '1d'
LSTM_LAYERS_N = 1
LSTM_INPUT_SIZE = 10
LEVERAGE_LEVEL_N = 5
LEVERAGE_DELTAS = (-1/LEVERAGE_LEVEL_N, 0, +1/LEVERAGE_LEVEL_N, )


# hyper params
LR = 1e-3
HIDDEN_SIZE = 128
LSTM_LAYERS_N = 1
LSTM_HIDDEN_SIZE = 16
BATCH_NORM = True
DROPOUT = 0.1
WEIGHT_DECAY = 0

# types
Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]
Reward = np.float32


#
# routine
#
def observe(my: Context[OhlcvWindow]) -> Obs:
    # feat 1
    ohlcv = my.event.win
    top = ohlcv.max()
    bottom = ohlcv.min()
    ohlcv_norm = (ohlcv - bottom) / (top - bottom)
    obs1 = ohlcv_norm[-N_FEATURE:].to_numpy()
    # feat 2
    ohlc = my.event.win.iloc[:, :-1]
    ohlc_pctl = price_percentile(ohlc)
    obs2 = ohlc_pctl[-N_FEATURE:].to_numpy()
    # feat 3
    obs3 = np.repeat(my.portfolio.leverage, len(obs1)).reshape(-1, 1)
    # obs
    obs = np.hstack((obs1, obs2, obs3)).astype(np.float32)
    return obs


def act(my: Context[OhlcvWindow], action: Action) -> float:
    delta = LEVERAGE_DELTAS[int(action)]
    target_weight = crop(my.portfolio.leverage + delta, 0, 1)
    my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
    return target_weight


def grant(my: Context[OhlcvWindow]) -> Reward:
    eq = my.portfolio.dashboard.equity
    # pr = my.memory['price'][INTERVAL]
    eq_r = np.float32(np.log(eq[-1] / eq[-INTERVAL]))
    # pr_r = np.float32(np.log(pr[-1] / pr[-INTERVAL]))
    # reward = eq_r - pr_r
    reward = eq_r
    return reward


def every(my: Context[OhlcvWindow]) -> None:
    my.memory['price'][INTERVAL].append(my.event.price)


#
# Env
#
class MyEnv(BasicTrEnv[Obs, Action]):
    # Env
    observation_space: Box = Box(low=0, high=1, shape=(N_FEATURE, ), )
    action_space: Discrete = Discrete(3)

    # Trader
    Market = YahooHistoricalWindows
    interval = INTERVAL
    symbol = SYMBOL
    length = LENGTH
    freq = FREQ

    @override
    def observe(self, my: Context[OhlcvWindow]) -> Obs:
        return observe(my)

    @override
    def act(self, my: Context[OhlcvWindow], action: Action) -> None:
        act(my, action)

    @override
    def grant(self, my: Context[OhlcvWindow]) -> Reward:
        return grant(my)

    @override
    def every(self, my: Context[OhlcvWindow]) -> None:
        every(my)


class TrainEnv(MyEnv):
    start = TRAIN_START
    end = TRAIN_END


class ValdEnv(MyEnv):
    start = VALD_START
    end = VALD_END


#
# Agent
#
class MyAgent(A2CDiscreteAgent[Obs, Action]):
    device = T.device('cuda')
    max_step = 500
    n_eps = 2500
    n_epoch = 100
    replay_size = 100*max_step
    batch_size = 128
    update_target_every = 10
    print_hash_every = 1
    rolling_reward_ma = 20
    report_progress_every = 25
    auto_save = True
    mean_reward_display_format = '+.6%'
    tensorboard = False
    gamma = 1

    def __init__(self) -> None:
        super().__init__()
        self.env = TrainEnv()
        self.vald_env = ValdEnv()
        net_kwargs: dict[str, Any] = dict(
            batch_norm=BATCH_NORM,
            dropout=DROPOUT,
            hidden_dim=HIDDEN_SIZE,
            lstm_layers_n=LSTM_LAYERS_N,
            lstm_input_dim=LSTM_INPUT_SIZE,
        )
        self.actor_critic_net = LSTM_ActorCriticDiscrete(self.obs_dim, self.action_dim, **net_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=LR,
                                    weight_decay=WEIGHT_DECAY)


#
# backtest
#
def benchmark_step(my: Context[OhlcvWindow]):
    if my.count.beginning:
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def agent_step(my: Context[OhlcvWindow]):
    every(my)
    if my.count.beginning:
        # starts with half position
        my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
    elif my.count.every(INTERVAL):
        # observe
        obs = observe(my)
        # take action
        action = agent.decide(obs)
        target_weight = act(my, action)
        # mark
        my.mark['price-pctl'] = price_percentile(my.event.win['Close'])[-1]
        my.mark['action'] = action.item()
        my.mark['target_weight'] = target_weight
        my.mark['reward'] = grant(my)
        my.mark['cum_reward'] = sum(my.mark['reward'])
    my.mark['price'] = my.event.price


def Env(name: str, do: Hook[OhlcvWindow]) -> Trader:
    return Trader(
        strategy=Strategy(name=name)
        .on(SYMBOL, OhlcvWindow, do=do),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS, start=TEST_START, end=TEST_END, length=LENGTH),
        broker=PaperEX(SYMBOLS)
    )


backtest = Backtest(
    Env('Benchmark', benchmark_step),
    Env('Agent', agent_step)
)

#
# main
#
agent = MyAgent()
agent.prompt()
backtest.run()
backtest.result.save()
