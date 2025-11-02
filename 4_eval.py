import torch
import ast
import numpy as np
import pandas as pd
from torch import nn
from two_model import impact_model

from three_training import (
    dtm_X_test, dtm_y_test, dtm_criterion,
    tfidf_X_test, tfidf_y_test, tfidf_criterion,
    curated_X_test, curated_y_test, curated_criterion
)

#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#load all 3 models

saved_dtm = torch.load('dtm_model.pth')
dtm_model = impact_model(input_size=saved_dtm['input_size'],
                         hidden_size=saved_dtm['hidden_size'],
                         output_size=saved_dtm['output_size'])
dtm_model.load_state_dict(saved_dtm['state_dict'])


saved_tfidf = torch.load('tfidf_model.pth')
tfidf_model = impact_model(input_size=saved_tfidf['input_size'],
                         hidden_size=saved_tfidf['hidden_size'],
                         output_size=saved_tfidf['output_size'])
tfidf_model.load_state_dict(saved_tfidf['state_dict'])

saved_curated = torch.load('curated_model.pth')
curated_model = impact_model(input_size=saved_curated['input_size'],
                         hidden_size=saved_curated['hidden_size'],
                         output_size=saved_curated['output_size'])
curated_model.load_state_dict(saved_curated['state_dict'])

#func to test models
def test(model, X_test, y_test, criterion):
  model.eval()
  with torch.inference_mode():
    y_test_logits = model(X_test)
    test_loss = criterion(y_test_logits, y_test)
    y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)

    acc = (y_test_preds == y_test).float().mean()

    return acc, test_loss

#send to device
dtm_model = dtm_model.to(device)
tfidf_model = tfidf_model.to(device)
curated_model = curated_model.to(device)
#testing X
dtm_X_test = dtm_X_test.to(device) 
tfidf_X_test = tfidf_X_test.to(device) 
curated_X_test = curated_X_test.to(device)
#testing y
dtm_y_test = dtm_y_test.to(device)
tfidf_y_test = tfidf_y_test.to(device)
curated_y_test = curated_y_test.to(device)

dtm_acc, dtm_test_loss = test(dtm_model, dtm_X_test, dtm_y_test, dtm_criterion)
print(f'dtm_acc: {dtm_acc} dtm_test_loss: {dtm_test_loss}\n')
tfidf_acc, tfidf_test_loss = test(tfidf_model, tfidf_X_test, tfidf_y_test, tfidf_criterion)
print(f'tfidf_acc: {tfidf_acc} tfidf_test_loss: {tfidf_test_loss}\n')
curated_acc, curated_test_loss = test(curated_model, curated_X_test, curated_y_test, curated_criterion)
print(f'curated_acc: {curated_acc} curated_test_loss: {curated_test_loss}\n')