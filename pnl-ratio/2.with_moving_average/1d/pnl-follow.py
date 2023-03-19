from typing import Literal, get_args

from pandas import Series
from trbox import Strategy, Trader
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy.context import Context

MaType = Literal['raw', 'sma', 'wma', ]

SYMBOL = 'SPY'
SYMBOLS = (SYMBOL, )
START = '2000-01-01'
END = None
FREQ = '1d'
LENGTH = [100, 200, ]
INTERVAL = [5, 21, ]
MA_TYPE = get_args(MaType)
MA_PERIOD = [21, 60]


def buy_hold(my: Context[OhlcvWindow]):
    if my.count.every(INTERVAL[-1]):
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def wma(win: Series):
    n = len(win)
    num = sum([(i+1)*v for i, v in enumerate(win)])
    denum = n * (n+1) / 2
    return num/denum


def pnl_ratio(
    win: Series,
    mode: MaType = 'raw',
    n: int = 5,
) -> float:
    pnlr = Series(win.rank(pct=True))
    match mode:
        case 'raw':
            return pnlr[-1]
        case 'sma':
            return pnlr[-n:].mean()
        case 'wma':
            return wma(pnlr[-n:])
        case _:
            return pnlr[-1]


def strategy(length: int, interval: int, ma_type: MaType, ma_period: int):
    def follow_pnl(my: Context[OhlcvWindow]):
        if my.count.every(interval):
            win = my.event.win['Close']
            weight = pnl_ratio(win, ma_type, ma_period)
            my.portfolio.rebalance(SYMBOL, weight, my.event.price)
            my.mark['pnlr-raw'] = pnl_ratio(win)
            my.mark[f'pnlr-sma{ma_period}'] = pnl_ratio(win, 'sma', ma_period)
            my.mark[f'pnlr-wma{ma_period}'] = pnl_ratio(win, 'wma', ma_period)
        my.mark['price'] = my.event.price

    return Trader(
        strategy=Strategy(
            name=f'L({length})-I({interval})-P({ma_type}{ma_period})')
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
        strategy(length, interval, ma_type, ma_period)
        for length in LENGTH
        for interval in INTERVAL
        for ma_type in MA_TYPE
        for ma_period in MA_PERIOD
    ],
)

bt.run(mode='process')

bt.result.save()
