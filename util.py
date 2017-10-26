""" Utility code."""

# Copyright (C) 2017  Wenchen Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import numpy as np


def symbol_to_path(symbol, base_dir=os.path.join(".", "yahoo_finance_data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price",save_image=False,save_dir="./"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not save_image:
      plt.show()
    else:
      plt.savefig(save_dir+title)
    plt.close()

def get_macd(group):

  def moving_average(group, n=9):
    sma = pd.rolling_mean(group, n)
    return sma

  def moving_average_convergence(group, nslow=26, nfast=12):
    emaslow = pd.Series.ewm(group, span=nslow, min_periods=1).mean()
    emafast = pd.Series.ewm(group, span=nfast, min_periods=1).mean()
    result = pd.DataFrame({'MACD': emafast - emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    return result

  return moving_average_convergence(group)


def Bollinger_Bands(stock_price, window_size, num_of_std):
  rolling_mean = stock_price.rolling(window=window_size).mean()
  rolling_std = stock_price.rolling(window=window_size).std()
  upper_band = rolling_mean + (rolling_std * num_of_std)
  lower_band = rolling_mean - (rolling_std * num_of_std)

  return rolling_mean, upper_band, lower_band


def Bollinger_Bands_given_sym_dates(sym, start_date,end_date,window_size=20, num_of_std=2):

  dates = pd.date_range(start_date - dt.timedelta(window_size*2-10), end_date) #TODO think better choose nan dates
  stock_price = get_data(sym, dates)

  rolling_mean, upper_band, lower_band = Bollinger_Bands(stock_price["SPY"], window_size,num_of_std)
  retrive_dates = pd.date_range(start_date, end_date)
  result = pd.DataFrame({'rolling_mean': rolling_mean, 'upper_band': upper_band, 'lower_band': lower_band},index=dates)
  result = result.dropna()
  return result


def momentum(sym,start_date,end_date,window_size=10):
  dates = pd.date_range(start_date - dt.timedelta(window_size), end_date)  # TODO think better choose nan dates
  stock_price = get_data(sym, dates)

  # print rolling_mean, upper_band,lower_band
  # M =
  result = pd.DataFrame()
  result = result.dropna()
  return result


def norm(l):
  l = np.array(l)
  return (l - l.min()) / (l.max() - l.min())

def save(object,file_path):
  with open(file_path,"wb") as handle:
    pickle.dump(object,handle)

def load(file_path):
  with open(file_path,"rb") as handle:
    obj = pickle.load(handle)
  return obj



if __name__=="__main__":
  #plot test
  sym = "GOOG"
  stdate = dt.datetime(2007, 1, 3)
  enddate = dt.datetime(2007, 12, 31)
  syms = [sym]
  dates = pd.date_range(stdate, enddate)
  prices_all = get_data(syms, dates)  # automatically adds SPY
  print prices_all
  # plot_data(prices_all)

  # test macd
  # record in panda format
  stdate = dt.datetime(2007, 1, 3)
  enddate = dt.datetime(2007, 12, 31)
  sym = ["GOOG"]
  dates = pd.date_range(stdate, enddate)
  prices_all = get_data(sym, dates)  # automatically adds SPY
  print get_macd(prices_all["SPY"])["MACD"].as_matrix()

  # test Bollinger band #TODO retrieve the first missing window data

  # print  Bollinger_Bands(prices_all["SPY"], 20, 2)
  bb = Bollinger_Bands_given_sym_dates(sym,stdate,enddate)
  print bb

  # momentum
  m = momentum(sym, stdate, enddate)
  print m