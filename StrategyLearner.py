"""
Q trader strategy learner. States are classified by kmeans now, Implementation 2017 Wenchen Li
"""

import os
import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import pickle
import numpy as np

CASH = "Cash"
STOCK = "Stock"

TRAIN_DIR = "./training_dir/"
kmeans_model_save_name = 'kmeans_model.pkl'
dyna_q_trader_model_save_name = "q_learner_tables.pkl"
training_record_save_name = "records.pkl"


class RawTradeFeatures(object):
    """
  get raw trade features like adjust closed price and volume of the trading
  """

    def __init__(self, symbol, sd, ed, ):
        self.syms = [symbol]
        self.dates = pd.date_range(sd, ed)

    def get_adj_price(self):
        prices_all = ut.get_data(self.syms, self.dates)  # automatically adds SPY
        prices = prices_all[self.syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later

        return prices, prices_SPY

    def get_vol(self):
        ## example use with new colname
        volume_all = ut.get_data(self.syms, self.dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[self.syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later

        return volume, volume_SPY


class StrategyLearner(object):
    """
  For the policy learning part:

  Select several technical features, and compute their values for the training data
  Discretize the values of the features
  Instantiate a Q-learner
  For each day in the training data:
    Compute the current state (including holding)
    Compute the reward for the last action
    Query the learner with the current state and reward to get an action
    Implement the action the learner returned (BUY, SELL, NOTHING), and update portfolio value
  """

    # constructor
    def __init__(self, verbose=True, save_dir=""):
        self.verbose = verbose
        self.Nothing = 0
        self.Buy = 1
        self.Sell = 2
        self.num_holding_state = 3  # 0 for 0, 1 for long, 2 for short
        self.num_feature_state = 100
        self.num_state = self.num_holding_state * self.num_feature_state
        self.num_action = 3
        self.shares_to_buy_or_sell = 100
        self.current_holding_state = 0  # prepare holding state, long, short, 0
        self.last_r = 0.0
        self.portfolio = {}
        self.num_epoch = 1000
        self.stop_threshold = 1.0
        self.epsilon = .01
        self.learner_dyan_iter = 0  # 200
        # keep records of each epoch and within each epoch the transaction
        self.records = []  # element for each epoch is (current_state,action,value, reward)

        self.negative_return_punish_factor = 1.3
        # save paths
        self.save_dir = save_dir
        self.current_working_dir = TRAIN_DIR + save_dir + "/"
        if not os.path.exists(self.current_working_dir):
            os.makedirs(self.current_working_dir)

    def get_current_portfolio_values(self, today_stock_price):
        return self.portfolio[CASH] + self.portfolio[STOCK] * today_stock_price

    def init_portfolio(self, sv,):
        self.portfolio[CASH] = float(sv)
        self.portfolio[STOCK] = 0

    def get_raw_data(self, symbol, sd, ed):
        # record in panda format
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        portfolio_value = prices_all[[symbol, ]]

        # Select several technical features, and compute their values for the training data

        tf = RawTradeFeatures(symbol, sd, ed)
        price, price_SPY = tf.get_adj_price()

        return dates,prices_all, trades, portfolio_value, tf, price, price_SPY

    def get_benchmark(self,symbol, price_SPY, prices_all,sv ):
        """
        get buy and hold benchmark result
        """
        # buy and hold the benchmark
        benchmark_price = price_SPY.as_matrix()
        benchmark_values = prices_all[[symbol, ]]  # only portfolio symbols
        benchmark_num_stock = int(sv / benchmark_price[0])
        cash = sv - float(benchmark_num_stock * benchmark_price[0])
        for i, p in enumerate(benchmark_price):
            benchmark_values.values[i, :] = cash + p * benchmark_num_stock

        return benchmark_values

    def get_derived_data(self,symbol, prices_all,sd,ed):
        """
        get derived data macd and bollinger band given the raw data
        """
        # macd
        macd = ut.norm(ut.get_macd(prices_all[symbol])["MACD"].as_matrix())

        # bollinger band
        bb = ut.Bollinger_Bands_given_sym_dates([symbol], sd, ed)
        bb_rm = ut.norm(bb['rolling_mean'].as_matrix())
        bb_ub = ut.norm(bb['upper_band'].as_matrix())
        bb_lb = ut.norm(bb['lower_band'].as_matrix())

        return macd, bb,bb_rm, bb_ub, bb_lb

    def get_finalized_input(self,price, symbol,macd):
        """
        reformatted the finalized input macd and price
        """
        # input to model or later process
        l = price[symbol].as_matrix()
        price_array = l.copy()

        x = macd.reshape((-1, 1))

        return price_array, x

    def perform_actions_update_trades_value(self,action,price_array,trades,i):
        """
        given action perform the action and record the trades value.
        """
        if action == 0:  # do nothing
            if self.verbose: print "do nothing"
        elif action == 1:  # buy
            if self.current_holding_state == 0 or self.current_holding_state == 2:  # holding nothing or short
                self.portfolio[CASH] -= self.shares_to_buy_or_sell * price_array[i]
                self.portfolio[STOCK] += self.shares_to_buy_or_sell
            elif self.current_holding_state == 1:  # long
                if self.verbose: print "buy but long already, nothing to do"

        else:  # action sell
            if self.current_holding_state == 0 or self.current_holding_state == 1:  # holding nothing or long
                self.portfolio[CASH] += self.shares_to_buy_or_sell * price_array[i]
                self.portfolio[STOCK] -= self.shares_to_buy_or_sell
            elif self.current_holding_state == 2:  # short
                if self.verbose: print "sell but short already, nothing to do"

        assert np.abs(self.portfolio[STOCK]) <= self.shares_to_buy_or_sell
        # update self.holding state
        if self.portfolio[STOCK] == self.shares_to_buy_or_sell:
            self.current_holding_state = 1
        elif self.portfolio[STOCK] == -self.shares_to_buy_or_sell:
            self.current_holding_state = 2
        elif self.portfolio[STOCK] == 0:
            self.current_holding_state = 0
        else:
            if self.verbose: print self.portfolio, "current portfolio is not valid"

        trades.values[i, :] = self.shares_to_buy_or_sell
        if action == 0:
            trades.values[i, :] *= 0
        elif action == 1:
            trades.values[i, :] *= 1
        else:
            trades.values[i, :] *= -1

        return trades

    def save_plot_and_model(self,benchmark_values, portfolio_value,trades,save_plot=False):
        """
        save trained model and plot result if save_plot set to True
        """
        # save transaction and portfolio image and training records
        benchmark_values = benchmark_values.rename(columns={'SPY': "benchmark"})
        portfolio_value = portfolio_value.rename(columns={'SPY': "q-learn-trader"})
        if save_plot:
            p_value_all = portfolio_value.join(benchmark_values)
            ut.plot_data(trades, title="transactions_train", ylabel="amount", save_image=True,
                         save_dir=self.current_working_dir)
            ut.plot_data(p_value_all, title="portfolio value_train", ylabel="USD", save_image=True,
                         save_dir=self.current_working_dir)
        self.learner.save_model(table_name=self.current_working_dir + dyna_q_trader_model_save_name)

    def addEvidence(self, symbol="SPY",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):
        """
        train q learner
        :param symbol:security symbol
        :param sd: start date
        :param ed: end data
        :param sv: start fund value
        :return: return of the trade
        """

        self.init_portfolio(sv)

        dates, prices_all, trades, portfolio_value, tf, price, price_SPY = self.get_raw_data(symbol, sd, ed)

        benchmark_values = self.get_benchmark( symbol, price_SPY, prices_all, sv)

        macd, bb, bb_rm, bb_ub, bb_lb = self.get_derived_data( symbol, prices_all, sd, ed)

        price_array, x = self.get_finalized_input( price, symbol, macd)

        # discretize
        kmeans_model = KMeans(n_clusters=self.num_feature_state, random_state=0, )
        kmeans = kmeans_model.fit(x)
        pickle.dump(kmeans_model, open(self.current_working_dir + kmeans_model_save_name, 'wb'))
        feature_states = kmeans.labels_

        # Instantiate a Q-learner
        self.learner = ql.QLearner(num_states=self.num_state, num_actions=self.num_action, rar=.5, alpha=.001,
                                   alpha_decay=.99, dyna=self.learner_dyan_iter)
        self.last_cumulated_return = 0.0
        self.cumulated_return_indicator = 0

        # value iteration
        for k in xrange(self.num_epoch):
            for i, s in enumerate(feature_states):
                if i == len(feature_states) - 1:
                    portfolio_value.values[i, :] = self.get_current_portfolio_values(price_array[-1])
                    continue  # skip last because 2nd day price
                # compute the current state(include holding)
                current_holding_state = self.current_holding_state
                current_feature_state = s
                current_state = self.num_feature_state * current_holding_state + current_feature_state

                # computer the last reward
                r = self.last_r
                if r < 0:  # punish negative reward
                    r *= self.negative_return_punish_factor
                # Query the learner with the current state and reward to get an action
                action = self.learner.query(current_state, r)

                # Implement the action the learner returned (BUY, SELL, NOTHING), and update portfolio value
                last_portfolio_value = self.get_current_portfolio_values(price_array[i])  # sum(self.portfolio_values)

                trades = self.perform_actions_update_trades_value(action, price_array, trades, i)

                portfolio_value.values[i, :] = self.get_current_portfolio_values(price_array[i])
                self.last_r = (self.get_current_portfolio_values(price_array[i + 1]) - last_portfolio_value) / float(sv)
                if self.verbose: print self.last_r

            if self.verbose: print "epoch", k, " current cumulated return:", self.get_current_portfolio_values(
                price_array[-1]) / float(sv) - 1.0, "portfolio:", self.portfolio
            self.cumulated_return_indicator = self.last_cumulated_return / (
                self.get_current_portfolio_values(price_array[-1]) / float(sv))
            self.last_cumulated_return = self.get_current_portfolio_values(price_array[-1]) / float(sv) - 1

            if self.num_epoch - 1 != k:
                # rest portfolio
                self.portfolio[CASH] = sv
                self.portfolio[STOCK] = 0
                self.current_holding_state = 0
                self.last_r = 0.0
                # decay alpha
                self.learner.decay_alpha()

        self.save_plot_and_model(benchmark_values, portfolio_value, trades)

        trade_return = self.get_current_portfolio_values(price_array[-1]) / sv - 1.0
        return trade_return

    # this method should use the existing policy and test it against new data


    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):
        """
        test q learner
        :param symbol:security symbol
        :param sd: start date
        :param ed: end data
        :param sv: start fund value
        :return: trades: record of the trades, trade_return: return of the trade
        """
        self.init_portfolio(sv)

        dates, prices_all, trades, portfolio_value, tf, price, price_SPY = self.get_raw_data(symbol, sd, ed)

        benchmark_values = self.get_benchmark(symbol, price_SPY, prices_all, sv)

        macd, bb, bb_rm, bb_ub, bb_lb = self.get_derived_data( symbol, prices_all, sd, ed)

        price_array, x = self.get_finalized_input( price, symbol, macd)

        # kmeans model load
        kmeans_model = pickle.load(open(self.current_working_dir + kmeans_model_save_name, 'rb'))
        kmeans = kmeans_model.predict(x)
        feature_states = kmeans

        # Instantiate a Q-learner
        self.learner = ql.QLearner(num_states=self.num_state, num_actions=self.num_action)
        self.last_cumulated_return = 0.0
        self.cumulated_return_indicator = 0
        # load trained q table(if dyna,t_table and r_table)
        self.learner.load_model(table_name=self.current_working_dir + dyna_q_trader_model_save_name)

        # value iteration
        for i, s in enumerate(feature_states):
            if i == len(feature_states) - 1:
                portfolio_value.values[i, :] = self.get_current_portfolio_values(price_array[-1])
                continue  # skip last because 2nd day price

            # compute the current state(include holding)
            current_holding_state = self.current_holding_state
            current_feature_state = s
            current_state = self.num_feature_state * current_holding_state + current_feature_state

            # Query the learner with the current state
            action = self.learner.querysetstate(current_state)

            # Implement the action the learner returned (BUY, SELL, NOTHING), and update portfolio value
            last_portfolio_value = self.get_current_portfolio_values(price_array[i])  # sum(self.portfolio_values)

            trades = self.perform_actions_update_trades_value(action, price_array, trades, i)

            portfolio_value.values[i, :] = self.get_current_portfolio_values(price_array[i])

            self.last_r = (self.get_current_portfolio_values(price_array[i + 1]) - last_portfolio_value) / float(sv)
            if self.verbose: print self.last_r

        self.last_cumulated_return = self.get_current_portfolio_values(price_array[-1]) / float(sv) - 1

        self.save_plot_and_model(benchmark_values, portfolio_value, trades)

        trade_return = self.get_current_portfolio_values(price_array[-1]) / sv - 1.0
        if self.verbose: print "cumulated return=:", trade_return
        return trades, trade_return
