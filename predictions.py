import pandas as pd
import numpy as np

from os.path import isfile

from models.learner.meta_learner import MetaLearner

def predict_file(filename: str, meta_learner: MetaLearner, verbose: bool = False):
  
  if verbose:
    print("[INFO] Predicting the file provided using the Meta-Learner model trained or loaded")
  
  if not isfile(filename):
    raise FileNotFoundError("The file provided does not exist.")
  
  dataframe = pd.read_csv(filename)

  if verbose:
    print("[INFO] File loaded successfully...")
    print("[INFO] information of the file loaded: ")
    print("\tName: ", filename)
    print("\tSize: ", dataframe.shape)
    print("\tColumns: ", dataframe.columns)
    print("\tHead: \n", dataframe.head(), "\n")
    
  if verbose:
    print("[INFO] Generating predictions from the weak learners...")
  
  weak_learners_predictions = []
  for weak_learner in meta_learner.weak_learners:
    weak_learners_predictions.append(weak_learner.predict(dataframe["text"].to_list()))

    if verbose:
      print("\tWeak learner: ", weak_learner.model_name)
      print("\tPredictions shape: ", weak_learners_predictions[-1].shape)
      print()

  if verbose:
    print("[INFO] Total of weak learners predictions generated: ", len(weak_learners_predictions))
    print("[INFO] Shape predictions generated: ", np.array(weak_learners_predictions).shape, "\n")
    print("[INFO] Embeddings generated successfully...")

    print("[INFO] Generate the meta learner predictionå...")

  prediction = meta_learner.predict(weak_learners_predictions)

  # Insert the prediction column in the dataframe
  dataframe.insert(1, "prediction", [pred.tolist() for pred in prediction], True)

  if verbose:
    print("[INFO] Prediction generated successfully...")
    print("[INFO] Displaying the results: \n")

  print(dataframe, "\n")
  # TODO: save the dataframe into CSV file output
  # TODO: have a better output format for the dataframe

  # print(dataframe["text"].to_list())

