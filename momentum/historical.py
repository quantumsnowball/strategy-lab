from trbox import Strategy, Trader
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy.context import Context


def main():
    SYMBOL = 'BTC-USD'
    SYMBOLS = ('BTC-USD', )
    START = '2021-07-01'
    END = '2021-12-31'
    LENGTH = 30

    # on_window
    def rebalance(pct_target: float):
        def routine(my: Context[OhlcvWindow]):
            if my.count.every(30):
                my.portfolio.rebalance(SYMBOL, pct_target, my.event.price)
        return routine

    bt = Backtest(
        Trader(
            strategy=Strategy(name='Benchmark')
            .on(SYMBOL, OhlcvWindow, do=rebalance(1)),
            market=YahooHistoricalWindows(
                symbols=SYMBOLS,
                start=START,
                end=END,
                length=LENGTH),
            broker=PaperEX(SYMBOL)),
        Trader(
            strategy=Strategy(name='basic')
            .on(SYMBOL, OhlcvWindow, do=rebalance(0.5)),
            market=YahooHistoricalWindows(
                symbols=SYMBOLS,
                start=START,
                end=END,
                length=LENGTH),
            broker=PaperEX(SYMBOL))
    )

    print('Started backtest')
    bt.run(parallel=False)
    print('Finished backtest')

    bt.result.save()


if __name__ == '__main__':
    main()
