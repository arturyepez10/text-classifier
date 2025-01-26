import numpy as np

class Dataset:
  def __init__(self, train: str, file_type: str, text_column: str, label_columns, test: str = None, sep: str = ","):
    self.train = train
    self.test = test if test != None else train
    self.file_type = file_type
    self.sep = sep
    self.text_column = text_column
    self.label_column = np.array(label_columns)

  def __str__(self):
    return f"Train: {self.train}\n\tTest: {self.test}\n\tFile Type: {self.file_type}\n\tSeparator: {self.sep}\n\tText Column: {self.text_column}\n\tLabel Column: {self.label_column}"