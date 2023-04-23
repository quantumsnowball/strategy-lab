from ta.volatility import BollingerBands

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


def bband_follow(my: Context[OhlcvWindow]):
    my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)
    win = my.event.win['Close']
    bband = BollingerBands(win)
    my.mark['price'] = price = my.event.price
    my.mark['mavg'] = bband.bollinger_mavg().iloc[-1]
    my.mark['hband'] = hband = bband.bollinger_hband().iloc[-1]
    my.mark['lband'] = lband = bband.bollinger_lband().iloc[-1]
    my.mark['bbratio'] = bbratio = max(min((price-lband)/(hband-lband), 1), 0)
    if my.count.every(INTERVAL[0]):
        my.portfolio.rebalance(SYMBOL, bbratio, my.event.price)


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
        strategy=Strategy(name=f'bband-follow')
        .on(SYMBOL, OhlcvWindow, do=bband_follow),
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL))
)

bt.run(parallel=True)

bt.result.save()
