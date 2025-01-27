import pandas as pd
import numpy as np

from os.path import isfile

class Dataset:
  version = "0.1.0"
  
  def __init__(self):
    self._clear()

  def _clear(self):
    self.dataframe = pd.DataFrame()
    self.dataframe_processed = pd.DataFrame()
    self.name = ""
    self.file_path = ""

  def verify_file(self, path: str) -> bool:
    return isfile(path)
  
  def load(self, path: str, sep: str = "\t"):
    if not self.verify_file(path):
      raise FileNotFoundError(f"[ERROR] File '{path}' not found for loading Dataset.")
    
    self.dataframe = pd.read_csv(path, sep=sep)
    self.name = path.split("/")[-1]
    self.file_path = path

  def pre_process(self):
    """Dataset loaded into the class instance is pre-processed in order to be ready
    for the training process.

    In this version of the method, we make some assumptions on the dataset structure:
      - The first column is an identifier
      - The second column is the text to be classified
      - The third column and beyond are the labels for the text

    For the labels, we assume that their columns are binary values (0 or 1).

    The pre-processing will:
      - Separate the text from the labels
      - Convert the labels to a numpy array (of size as many as labels are originally)
    """
    # Obtain the text and the labels separately
    text = self.dataframe.iloc[:, 1:2]
    labels = self.dataframe.iloc[:, 2:].to_numpy()

    # Join the text and the labels in a new dataframe
    text_and_labels = [[text.iloc[index, 0], labels[index].tolist()] for index in range(0,int(labels.shape[0]))]
    
    self.dataframe_processed = pd.DataFrame(text_and_labels, columns=["text", "target"])
