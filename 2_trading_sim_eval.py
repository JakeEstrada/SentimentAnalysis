import pandas as pd
import numpy as np
import torch as T
import ast
import math
from torcheval.metrics.functional import r2_score as R2Score
from torch.nn import MSELoss
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
import importlib
from torch import nn

class impact_model(nn.Module):
  def __init__(self, input_size, hidden_size=16, output_size=7):
    super().__init__()

    self.layers = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(0.7),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(0.7),
                                nn.Linear(hidden_size, output_size))

  def forward(self, x):
    return self.layers(x)

buysell = importlib.import_module('1_trading_rules')
initial_money = 100000
current_money = initial_money
model_name = 'tfidf_model.pt'
dataset_name = 'tfidf_train.csv'



data_full = pd.read_csv(dataset_name).sort_values('date')
syms = data_full['symbol'].unique()
stock_nums = {}
for i in syms:
    stock_nums[i] = [0,0]
print(syms)
data_hist_full = pd.read_csv('historical_prices_impact_new.csv').sort_values('date')
print(len(data_hist_full))
data_hist_full = data_hist_full[data_hist_full['symbol'].isin(syms)]
print(len(data_hist_full))
data_hist = data_hist_full.to_numpy()
data_eval = np.array([np.array(ast.literal_eval(x), dtype=float) for x in data_full['news_vector']])
data_eval_y =data_full['impact_score'].values
data_eval_y = T.LongTensor(data_eval_y)
data_eval_t = T.FloatTensor(data_eval)
data_head = pd.read_csv('analyst_ratings.csv').sort_values('date')
data_head = data_head[data_head['stock'].isin(syms)]
print(len(data_head))
hidden_size = 8
output_size = 7
model = impact_model(input_size=data_eval_t.shape[1],hidden_size=hidden_size, output_size=output_size)
model.load_state_dict(T.load(model_name, weights_only=True))
model.eval()
with T.inference_mode():
    transaction_date = []
    symbol = []
    trade_type = []
    number_shares = []
    price = []
    transaction_amnt = []
    cash_after_trade = []
    headline = []
    impact_score = []
    rows = []
    curr_year = data_eval[0][0]
    annual_money = []
    end_i = 0
    beginning_i = 0
    logits = model(data_eval_t)
    preds = T.softmax(logits, dim = 1).argmax(dim = 1)
    print(len(preds))
    for i in range(0,len(preds)+1):
        if i == len(preds):
            print(stock_nums)
            for j in stock_nums:
                current_money += stock_nums[j][0]*stock_nums[j][1]
        else:
            transaction_date.append(data_hist[i][0])
            symbol.append(data_hist[i][1])
            price.append(data_hist[i][2])
            curr_date = data_hist[i][0].split('-')
            impact_score.append(data_hist[i][-1])
            hd = data_head[(data_head['date'].str.contains(data_full['date'][i])) & (data_head['stock'] == data_full['symbol'][i])]['headline'].values
            # hd = data_head[(data_head['date'] == data_eval[i][0])]
            # hd = hd[(data_head['stock'] == data_eval[i][1])]
            # hd = hd['headline'].values
            headline.append(hd)
            print(preds[i])
            if curr_date[0] != curr_year:
                annual_money.append( sum(transaction_amnt[beginning_i:end_i]))
                beginning_i = i
            if preds[i] > 0:
                shares = buysell.buy_rule(data_hist[i][2], current_money, data_hist[i][11], preds[i])
                stock_nums [symbol[i]] = [stock_nums[symbol[i]][0]+shares,data_hist[i][2]]
                number_shares.append(shares)
                transaction_amnt.append(number_shares[i]*data_hist[i][2])
                cash_after_trade.append(current_money-number_shares[i]*data_hist[i][2])
                trade_type.append(0)
                current_money -= number_shares[i]*data_hist[i][2]
            elif preds[i] < 0:
                shares = buysell.sell_rule(data_hist[i][2], data_hist[i][11], preds[i])
                stock_nums [symbol[i]] = [stock_nums[symbol[i]][0]-shares,data_hist[i][2]]
                number_shares.append(stock_nums[symbol[i]])
                transaction_amnt.append(stock_nums[symbol[i]] * data_hist[i][2])
                cash_after_trade.append(current_money + stock_nums[symbol[i]] * data_hist[i][2])
                current_money += stock_nums[symbol[i]] * data_hist[i][2]
                trade_type.append(1)

            elif preds[i] == 0:
                number_shares.append(0)
                transaction_amnt.append(0)
                cash_after_trade.append(current_money)
                trade_type.append(2)
            print(i)
            end_i = i

    end_money = current_money - initial_money
    average_annual = (((sum(annual_money) / len(annual_money)) - initial_money)/initial_money)*100
    percent_return = (end_money / initial_money)*100
    final_balance = current_money
    print(end_money)
    print(average_annual)
    print(percent_return)
    print(final_balance)
    final_df = pd.DataFrame()
    final_df['total_gain'] = [end_money]
    final_df['average_annual'] = [average_annual]
    final_df['total_return'] = [percent_return]
    final_df['balance'] = [final_balance]
    final_df.to_csv('final_summary.csv', index=False)
    trade_log = pd.DataFrame()
    trade_log['transaction_date'] = transaction_date
    trade_log['symbol'] = symbol
    trade_log['trade_type'] = trade_type
    trade_log['number_shares'] = number_shares
    trade_log['price'] = price
    trade_log['cash_after_trade'] = cash_after_trade
    trade_log['headline'] = headline
    trade_log['pred_impact'] = preds
    trade_log['impact_score'] = impact_score
    trade_log.to_csv('trade_log.csv', index=False)