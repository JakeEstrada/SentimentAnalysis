import torch
import ast
import numpy as np
import pandas as pd
from torch import nn
from two_model import setup_model

#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Loading csv files
#training csvs
dtm_train_df = pd.read_csv('dtm_train.csv')
tfidf_train_df = pd.read_csv('tfidf_train.csv')
curated_train_df = pd.read_csv('curated_train.csv')
#test csvs
dtm_test_df = pd.read_csv('dtm_test.csv')
tfidf_test_df = pd.read_csv('tfidf_test.csv')
curated_test_df = pd.read_csv('curated_test.csv')

#convert impact scores to positive mapping 0-6 instad of -3-3
def map(df):
    df['impact_score'] = df['impact_score'] + 3
    df.head(10)
    return df

#map all dfs
dtm_train_df = map(dtm_train_df)
tfidf_train_df = map(tfidf_train_df)
curated_train_df = map(curated_train_df)
dtm_test_df = map(dtm_test_df)
tfidf_test_df = map(tfidf_test_df)
curated_test_df = map(curated_test_df)

#get X and y for each csv
#training X convert news_vector column into python list
dtm_X_train = np.array([np.array(ast.literal_eval(x), dtype=float) for x in dtm_train_df['news_vector']])
tfidf_X_train = np.array([np.array(ast.literal_eval(x), dtype=float) for x in tfidf_train_df['news_vector']])
curated_X_train = np.array([np.array(ast.literal_eval(x), dtype=float) for x in curated_train_df['news_vector']])

#training y
dtm_y_train = dtm_train_df['impact_score'].values
tfidf_y_train = tfidf_train_df['impact_score'].values
curated_y_train = curated_train_df['impact_score'].values

#testing X convert news_vector column into python list
dtm_X_test = np.array([np.array(ast.literal_eval(x), dtype=float) for x in dtm_test_df['news_vector']])
tfidf_X_test = np.array([np.array(ast.literal_eval(x), dtype=float) for x in tfidf_test_df['news_vector']])
curated_X_test = np.array([np.array(ast.literal_eval(x), dtype=float) for x in curated_test_df['news_vector']])

#testing y
dtm_y_test = dtm_test_df['impact_score'].values
tfidf_y_test = tfidf_test_df['impact_score'].values
curated_y_test = curated_test_df['impact_score'].values

#convert X's and y's into tensors
#might need to do = torch.FloatTensor(X/y)
#training X
dtm_X_train = torch.FloatTensor(dtm_X_train)
tfidf_X_train = torch.FloatTensor(tfidf_X_train)
curated_X_train = torch.FloatTensor(curated_X_train)

#training y
dtm_y_train = torch.LongTensor(dtm_y_train)
tfidf_y_train = torch.LongTensor(tfidf_y_train)
curated_y_train = torch.LongTensor(curated_y_train)

#testing X
dtm_X_test = torch.FloatTensor(dtm_X_test)
tfidf_X_test = torch.FloatTensor(tfidf_X_test)
curated_X_test = torch.FloatTensor(curated_X_test)

#testing y
dtm_y_test = torch.LongTensor(dtm_y_test)
tfidf_y_test = torch.LongTensor(tfidf_y_test)
curated_y_test = torch.LongTensor(curated_y_test)


dtm_input_size = dtm_X_train.shape[1]
tfidf_input_size = tfidf_X_train.shape[1]
curated_input_size = curated_X_train.shape[1]

dtm_hidden_size = 7
tfidf_hidden_size = 7
curated_hidden_size = 7

output_size = 7

#setup models
#input shape of X for dtm
dtm_model, dtm_criterion, dtm_optimizer = setup_model(dtm_input_size,dtm_hidden_size,output_size)

tfidf_model, tfidf_criterion, tfidf_optimizer = setup_model(tfidf_input_size, tfidf_hidden_size,output_size)

curated_model, curated_criterion, curated_optimizer = setup_model(curated_input_size,curated_hidden_size,output_size)


#func to train models
def train(model, X_train, y_train, criterion, optimizer, epochs):

  for epoch in range(epochs):
    model.train()
    y_logits = model(X_train)
    loss = criterion(y_logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      model.eval()
      with torch.inference_mode():
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        acc = (y_pred == y_train).float().mean()
        print(f'on epoch {epoch}, loss:{loss:.3f}, acc: {acc:.3f}')

  return model

if __name__ == '__main__':
  #send to device
  dtm_model = dtm_model.to(device)
  tfidf_model = tfidf_model.to(device)
  curated_model = curated_model.to(device)

  dtm_X_train = dtm_X_train.to(device) 
  tfidf_X_train = tfidf_X_train.to(device) 
  curated_X_train = curated_X_train.to(device)

  #training y
  dtm_y_train = dtm_y_train.to(device) 
  tfidf_y_train = tfidf_y_train.to(device) 
  curated_y_train = curated_y_train.to(device)

  #testing X
  dtm_X_test = dtm_X_test.to(device) 
  tfidf_X_test = tfidf_X_test.to(device) 
  curated_X_test = curated_X_test.to(device)

  #testing y
  dtm_y_test = dtm_y_test.to(device)
  tfidf_y_test = tfidf_y_test.to(device)
  curated_y_test = curated_y_test.to(device)
  
  
  #train models
  epochs=300
  print('training dtm model')
  dtm_model = train(dtm_model, dtm_X_train, dtm_y_train, dtm_criterion, dtm_optimizer, epochs)
  print('training tfidf model')
  tfidf_model = train(tfidf_model, tfidf_X_train, tfidf_y_train, tfidf_criterion, tfidf_optimizer, epochs)
  print('training curated model')
  curated_model = train(curated_model, curated_X_train, curated_y_train, curated_criterion, curated_optimizer, epochs)


  dtm_name = 'dtm_model.pth'
  tfidf_name = 'tfidf_model.pth'
  curated_name = 'curated_model.pth'

  torch.save({'state_dict': dtm_model.state_dict(),
            'input_size': dtm_input_size,
            'hidden_size': dtm_hidden_size,
            'output_size': output_size
            }, dtm_name)


  torch.save({'state_dict': tfidf_model.state_dict(),
            'input_size': tfidf_input_size,
            'hidden_size': tfidf_hidden_size,
            'output_size': output_size
            }, tfidf_name)

  torch.save({'state_dict': curated_model.state_dict(),
            'input_size': curated_input_size,
            'hidden_size': curated_hidden_size,
            'output_size': output_size
            }, curated_name)
