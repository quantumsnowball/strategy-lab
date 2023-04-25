import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box
from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet
from mlbox.trenv import BasicTrEnv
from mlbox.utils import crop, pnl_ratio
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.binance.historical.windows import BinanceHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.strategy.types import Hook
from trbox.trader import Trader
from typing_extensions import override

# train
TRAIN_START = '2022-07-01'
TRAIN_END = '2022-09-30'
# validation
VALD_START = '2022-10-01'
VALD_END = '2022-12-31'
# test
TEST_START = '2023-01-01'
TEST_END = '2023-03-31'

SYMBOL = 'BTCUSDT'
SYMBOLS = (SYMBOL, )
LENGTH = 200
INTERVAL = 5
FREQ = '1h'
STEP = 0.2
START_LV = 0.01
N_FEATURE = 150
MODEL_NAME = 'model.pth'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]
Reward = np.float32


#
# routine
#
def observe(my: Context[OhlcvWindow]) -> Obs:
    win = my.event.win['Close']
    pnlr = pnl_ratio(win)
    obs = np.array(pnlr[-N_FEATURE:], dtype=np.float32)
    return obs


def act(my: Context[OhlcvWindow], action: Action) -> float:
    target_weight = crop(action.item(), low=0, high=1)
    my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
    return target_weight


def grant(my: Context[OhlcvWindow]) -> Reward:
    eq = my.portfolio.dashboard.equity
    pr = my.memory['price'][INTERVAL]
    eq_r = np.float32(np.log(eq[-1] / eq[-INTERVAL]))
    pr_r = np.float32(np.log(pr[-1] / pr[-INTERVAL]))
    reward = eq_r - pr_r
    return reward


def every(my: Context[OhlcvWindow]) -> None:
    my.memory['price'][INTERVAL].append(my.event.price)


#
# Env
#
class MyEnv(BasicTrEnv[Obs, Action]):
    # Env
    observation_space: Box = Box(low=0, high=1, shape=(N_FEATURE, ), )
    action_space: Box = Box(low=0, high=1, shape=(1, ), )

    # Trader
    Market = BinanceHistoricalWindows
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
class MyAgent(DDPGAgent[Obs, Action]):
    device = T.device('cuda')
    max_step = 500
    n_eps = 5000
    n_epoch = 2
    replay_size = 100*max_step
    batch_size = 256
    update_target_every = 10
    print_hash_every = 5
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
        self.min_noise = 0.2
        self.max_noise = self.max_action * 1.0
        self.actor_net = DDPGActorNet(self.obs_dim, self.action_dim,
                                      min_action=self.min_action,
                                      max_action=self.max_action).to(self.device)
        self.actor_net_target = DDPGActorNet(self.obs_dim, self.action_dim,
                                             min_action=self.min_action,
                                             max_action=self.max_action).to(self.device)
        self.critic_net = DDPGCriticNet(self.obs_dim, self.action_dim).to(self.device)
        self.critic_net_target = DDPGCriticNet(self.obs_dim, self.action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-3)


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
        my.mark['pnlr'] = pnl_ratio(my.event.win['Close'])[-1]
        my.mark['action'] = action.item()
        my.mark['target_weight'] = target_weight
        my.mark['reward'] = grant(my)
        my.mark['cum_reward'] = sum(my.mark['reward'])
    my.mark['price'] = my.event.price


def Env(name: str, do: Hook[OhlcvWindow]) -> Trader:
    return Trader(
        strategy=Strategy(name=name)
        .on(SYMBOL, OhlcvWindow, do=do),
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS, start=TEST_START, end=TEST_END, length=LENGTH, freq=FREQ),
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
agent.prompt(MODEL_NAME)
backtest.run()
backtest.result.save()
