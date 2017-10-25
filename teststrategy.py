"""
Test a Strategy Learner.  (c) 2017 Wenchen Li
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl

def test_code(train_q=None,out_q=None,verb = True, test_only=False,save_dir=".",):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb,save_dir=save_dir)

    if not test_only:
        # set parameters for training the learner
        sym = "SPY"
        stdate =dt.datetime(2016,4,30)
        enddate =dt.datetime(2017,1,3) # just a few days for "shake out"

        # train the learner
        trade_return_train = learner.addEvidence(symbol = sym, sd = stdate,
            ed = enddate, sv = 10000)
        if train_q:
            train_q.put(trade_return_train)
    # set parameters for testing
    sym = "SPY"
    stdate =dt.datetime(2017,1,3)
    enddate =dt.datetime(2017,9,30)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    # if verb: print prices

    # test the learner
    df_trades,trade_return = learner.testPolicy(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000)
    if out_q:
        out_q.put(trade_return)
    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"

if __name__=="__main__":
    from multiprocessing import Process,Queue
    import numpy as np

    train_q = Queue()
    out_q = Queue()
    train_resultdict = {}
    test_resultdict = {}
    total_num_simulation_left = 1000

    while total_num_simulation_left>0:

        nprocs_each_iter = 10
        procs = []
        current_training_ids = []
        for i in xrange(nprocs_each_iter):
            training_id = total_num_simulation_left - i
            current_training_ids.append(training_id)
            proc = Process(target=test_code,args=(train_q,out_q,False,False,str(training_id),))
            procs.append(proc)
            proc.start()

        for training_id in current_training_ids:
            train_resultdict[training_id] = train_q.get()
            test_resultdict[training_id]= out_q.get()

        for p in procs:
            p.join()

        total_num_simulation_left -= nprocs_each_iter

        print "train:"
        print "each strategy return:", train_resultdict.values()
        print "mean return:", np.average(train_resultdict.values())
        print "std return:", np.std(train_resultdict.values())

    mean = np.average(test_resultdict.values())
    std = np.std(test_resultdict.values())
    yearly_risk_free_rate = .05 #  https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2007
    print "test:"
    print "each strategy return:",test_resultdict.values()
    print "mean return:",mean
    print "std return:",std
    print "max, min return:",max(test_resultdict.values()), min(test_resultdict.values())
    print "sharpe ratio:", (mean - yearly_risk_free_rate)/std



