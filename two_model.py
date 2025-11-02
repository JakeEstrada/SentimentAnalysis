import torch
import ast
import numpy as np
import pandas as pd
from torch import nn

#Class for each MLP
class impact_model(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=7):
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

#setup model function
def setup_model(input_size, hidden_size, output_size):
  model = impact_model(input_size, hidden_size, output_size)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, weight_decay=1e-4)

  return (model, criterion, optimizer)
