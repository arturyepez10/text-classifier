import pandas as pd
import numpy as np

from os.path import isfile

from models.learner.meta_learner import MetaLearner
from models.dataset import Dataset
from models.parameters import Parameters

def predict_file(params: Parameters, filename: str, meta_learner: MetaLearner, verbose: bool = False):
  
  if verbose:
    print("[INFO] Predicting the file provided using the Meta-Learner model trained or loaded")
  
  if not isfile(filename):
    raise FileNotFoundError("The file provided does not exist.")
  
  dataset = Dataset()
  dataset.load(filename)

  if verbose:
    print("[INFO] File loaded successfully...")
    print("[INFO] information of the file loaded: ")
    print("\tName: ", filename)
    print("\tSize: ", dataset.dataframe.shape)
    print("\tColumns: ", dataset.dataframe.columns)
    print("\tHead: \n", dataset.dataframe.head(), "\n")
    
  if verbose:
    print("[INFO] Generating predictions from the weak learners...")
  
  weak_learners_predictions = []
  for weak_learner in meta_learner.weak_learners:
    weak_learners_predictions.append(weak_learner.predict(dataset.dataframe["text"].to_list()))

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
  dataset.append_target(prediction)

  if verbose:
    print("[INFO] Prediction generated successfully...")

    print("[INFO] Processing the results to enhance readability...")

  dataset.revert_processing(params.dataset.label_column)

  if verbose:
    print("\n[INFO] Displaying the results: \n")

  print(dataset.dataframe)

  if verbose:
    print("\n[INFO] Results displayed successfully...")
    print("[INFO] Predictions being saved as CSV file...")

  dataset.save("non-processed")

  if verbose:
    print("[INFO] Predictions saved successfully...")
    print("[INFO] Predictions process finished...")