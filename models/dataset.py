import pandas as pd

from os import path as pathLib, mkdir

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
    return pathLib.isfile(path)
  
  def load(self, path: str, sep: str = "\t"):
    if not self.verify_file(path):
      raise FileNotFoundError(f"[ERROR] File '{path}' not found for loading Dataset.")
    
    self.dataframe = pd.read_csv(path, sep=sep)
    self.name = path.split("/")[-1]
    self.file_path = path
    self.emotion_labels = self.dataframe.columns[2:].tolist()

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

  def append_target(self, prediction):
    """Generates a "target" column with the predictions to the dataset in order to save the predictions
    in the format of the dataset pre-processed.

    Result is the same as the function `pre_process` and store in the variable `dataframe_processed`.

    To ensure that there is consistency, we are assuming that the column text is the only column in the 
    `dataframe` attribute.
    """
    # Join the text and the labels in a the processed dataframe variable
    self.dataframe_processed = self.dataframe.copy()
    self.dataframe_processed.insert(1, "target", [pred.tolist() for pred in prediction], True)

  def revert_processing(self, emotion_labels: list = []):
    """Reverts the pre-process done to the dataset in order to save the predictions
    in the original format of a dataset without processing.

    Result is the inverse as the function `pre_process` and store in the variable `dataframe`.

    For this case, we just append
    """
    # Separate the text and the labels in a new dataframe
    text = self.dataframe_processed["text"].tolist()
    labels = self.dataframe_processed["target"].tolist()

    # Join the text and the labels in a new dataframe
    text_and_labels = [[text[index], *labels[index]] for index in range(0,int(len(labels)))]

    self.dataframe = pd.DataFrame(text_and_labels, columns=["text", *emotion_labels])


  def save(self, dataframe: str, name: str = "", sep: str = "\t"):
    # We choose which dataframe we want to save
    dt: pd.DataFrame = None

    if dataframe == "processed":
      dt = self.dataframe_processed
    
    if dataframe == "non-processed":
      dt = self.dataframe

    if dt is None:
      raise ValueError("The dataframe to save is not defined.")
    
    # We choose the name of the file and verify the output folder's existence
    filename = name

    if not filename:
      filename = "predictions_" + self.name + ".csv"

    if not pathLib.exists("out/"):
      mkdir("out/")

    # Save the dataframe as a CSV file using the same Pandas helpers
    dt.to_csv("out/" + filename, sep=sep, index=False)
    