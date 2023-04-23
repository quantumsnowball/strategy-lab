from pandas import Series
from ta.momentum import RSIIndicator

from trbox import Strategy, Trader
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.binance.historical.windows import BinanceHistoricalWindows
from trbox.strategy.context import Context

SYMBOL = 'BTCUSDT'
SYMBOLS = (SYMBOL, )
START = '2023-01-01'
END = None
FREQ = '1h'
LENGTH = [200, ]
INTERVAL = [24, ]


def buy_hold(my: Context[OhlcvWindow]):
    if my.count.every(INTERVAL[-1]):
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def rsi2_follow(my: Context[OhlcvWindow]):
    my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)
    win = my.event.win['Close']

    my.mark['price'] = my.event.price
    my.mark['rsi2'] = RSIIndicator(win, 2).rsi().iloc[-1]
    my.mark['rsi14'] = RSIIndicator(win, 14).rsi().iloc[-1]


bt = Backtest(
    Trader(
        strategy=Strategy(name='.buy-hold')
        .on(SYMBOL, OhlcvWindow, do=buy_hold),
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL)),
    Trader(
        strategy=Strategy(name=f'rsi-follow')
        .on(SYMBOL, OhlcvWindow, do=rsi2_follow),
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL))
)

print('Started backtest')
bt.run(parallel=True)
print('Finished backtest')

bt.result.save()
