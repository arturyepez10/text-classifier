import numpy as np

class Classifier:
  def __init__(self, hidden_layer_sizes, random_state: int, early_stopping: bool):
    self.hidden_layer_sizes = tuple(i for i in hidden_layer_sizes)
    self.random_state = random_state
    self.early_stopping = early_stopping

  def __str__(self):
    return f"\t\tHidden Layer Sizes: {self.hidden_layer_sizes}\n\t\tRandom State: {self.random_state}\n\t\tEarly Stopping: {self.early_stopping}"