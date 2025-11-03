import pandas as pd
import numpy as np
import torch as T
import math

def buy_rule(stock_price, current_cash, alpha_mult, impact):
    if (current_cash/stock_price) > 1:
        return max(1,math.floor(alpha_mult*impact*100/stock_price))
    return 0

def sell_rule(buy_rule_shares, owned_shares):
    if owned_shares >= buy_rule_shares:
        return owned_shares-buy_rule_shares
    return owned_shares