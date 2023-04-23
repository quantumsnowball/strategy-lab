from pandas import Series
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
LENGTH = [50, 100, 150, 200, 250, ]
INTERVAL = [12, 24, 48, ]


def buy_hold(my: Context[OhlcvWindow]):
    if my.count.every(INTERVAL[-1]):
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def pnl_ratio(win: Series) -> float:
    pnlr = Series(win.rank(pct=True))
    return pnlr[-1]


def strategy(length: int, interval: int):
    def follow_pnl(my: Context[OhlcvWindow]):
        if my.count.every(interval):
            win = my.event.win['Close']
            pnlr = pnl_ratio(win)
            weight = pnlr
            my.portfolio.rebalance(SYMBOL, weight, my.event.price)
            my.mark['pnl-ratio'] = pnlr
        my.mark['price'] = my.event.price

    return Trader(
        strategy=Strategy(name=f'L({length})-I({interval})')
        .on(SYMBOL, OhlcvWindow, do=follow_pnl),
        market=BinanceHistoricalWindows(
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
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL)),
    *[
        strategy(length, interval)
        for length in LENGTH
        for interval in INTERVAL
    ]
)

bt.run(mode='process')

# bt.result.save()
