# Q-learning trader
A technical q trader on SPY given macd.

## run
`python teststratgy.py`

## structure
QLearner.py: an independent tabular (dyna)q-learner.

StrategyLearner.py: Build upon Qlearner.py to learn the trading strategy

testStrategy.py: train and test the StrategyLearner

util.py: some helper functions for the model

## detail on experiments
模型框架： q-learning ( model-free reinforcement learning)
模型输入： market factor like adjust close price, macd, bollinger band,
模型输出： 3种行为(long, short, do_nothing)
模型流程：
1. 输入 市场因子， 模型把市场因子分类成不同状态（state)
2. q-learing model ：
        训练阶段： 在给定输入的状态（state) ，搜索q_table并给出行为（long, short, do_nothing), 通过第二天的股价得到奖励(rewards), 然后根据bellman equation更新q_table
        测试阶段： 在给定输入的状态（state) ，搜索q_table并给出行为（long, short, do_nothing).

实验细节：
交易根据adjust_close_price 和其衍生计算结果（ macd）
每一个fund期初有10000， 交易限制为仅允许(100 long SPY, 100 short SPY, do nothing)三种行为。
实验根据1000次的模拟交易。
训练数据为第一年的后180天交易数据。
测试数据为第二年一整年的交易数据。

## experiment result

please see [this link](https://www.evernote.com/shard/s120/sh/0fa6db4e-6cc8-4e48-bf50-1a8909a2d1e6/30725de60925143d)





