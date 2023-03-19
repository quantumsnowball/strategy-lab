from typing import Literal, get_args

import numpy as np
from pandas import Series
from trbox import Strategy, Trader
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy.context import Context

SYMBOL = 'SPY'
SYMBOLS = (SYMBOL, )
START = '2000-01-01'
END = None
FREQ = '1d'
LENGTH = [100, 200, ]
INTERVAL = [5, 21, ]
SIGMOID_SCALE = [None, 10, 25]


def buy_hold(my: Context[OhlcvWindow]):
    if my.count.every(INTERVAL[-1]):
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def pnl_ratio(win: Series) -> float:
    pnlr = Series(win.rank(pct=True))
    return pnlr[-1]


def strategy(length: int, interval: int, sigmoid_scale: int | None):
    def follow_pnl(my: Context[OhlcvWindow]):
        if my.count.every(interval):
            win = my.event.win['Close']
            pnlr = pnl_ratio(win)
            if sigmoid_scale is not None:
                norm_pnlr = (pnlr-0.5)*sigmoid_scale
                weight = sigmoid(norm_pnlr)
                my.mark[f'pnlr-sigmoid{sigmoid_scale}'] = sigmoid(norm_pnlr)
            else:
                weight = pnlr
            my.portfolio.rebalance(SYMBOL, weight, my.event.price)
            my.mark['pnlr-raw'] = pnlr
        my.mark['price'] = my.event.price

    return Trader(
        strategy=Strategy(
            name=f'L({length})-I({interval})-Sigmoid({sigmoid_scale})')
        .on(SYMBOL, OhlcvWindow, do=follow_pnl),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=length),
        broker=PaperEX(SYMBOL))


bt = Backtest(
    Trader(
        strategy=Strategy(name='.buy-hold')
        .on(SYMBOL, OhlcvWindow, do=buy_hold),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL)),
    *[
        strategy(length, interval, sigmoid_scale)
        for length in LENGTH
        for interval in INTERVAL
        for sigmoid_scale in SIGMOID_SCALE
    ],
)

bt.run(mode='process')

bt.result.save()
