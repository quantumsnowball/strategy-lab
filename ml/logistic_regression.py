from sklearn.linear_model import LogisticRegression
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
LENGTH = [500, ]
INTERVAL = [24, ]


def buy_hold(my: Context[OhlcvWindow]):
    if my.count.every(INTERVAL[-1]):
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def ml_prob_follow(my: Context[OhlcvWindow]):
    if my.count.beginning:
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)
        win = my.event.win['Close']
        X = [r for r in win.pct_change().dropna().rolling(5)
             if len(r) >= 5][:-5]
        y = (win.pct_change(5).dropna() > 0).astype(float).shift(-5).dropna()
        clf = LogisticRegression().fit(X, y)
        setattr(my, 'clf', clf)

    if my.count.every(5):
        win = my.event.win['Close']
        X = [list(win.pct_change().iloc[-5:]), ]
        clf = getattr(my, 'clf')
        y = clf.predict_proba(X)
        weight = y[0][1]
        my.portfolio.rebalance(SYMBOL, weight, my.event.price)
        my.mark['y_prob'] = weight

    my.mark['price'] = my.event.price


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
        strategy=Strategy(name=f'ml-follow')
        .on(SYMBOL, OhlcvWindow, do=ml_prob_follow),
        market=BinanceHistoricalWindows(
            symbols=SYMBOLS,
            start=START,
            end=END,
            freq=FREQ,
            length=LENGTH[0]),
        broker=PaperEX(SYMBOL))
)

bt.run(mode='process')

bt.result.save()
