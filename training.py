from models.parameters import Parameters
from models.dataset import Dataset
from models.learner.weak_learner import WeakLearner
from models.learner.meta_learner import MetaLearner

def train_model(path: str, verbose: bool = False):
  # Load the parameters from the file path passed as argument to the execution
  params = Parameters.from_json(path)

  if verbose:
    print("[INFO] This were the params loaded from the path provided: \n", params, "\n")

  # Load the dataset from the path provided in the parameters file
  dataset = Dataset()
  dataset.load(params.dataset.train)

  if verbose:
    print("[INFO] information of the dataset loaded: ")
    print("\tName: ", dataset.name)
    print("\tSize: ", dataset.dataframe.shape)
    print("\tColumns: ", dataset.dataframe.columns)
    print("\tHead: \n", dataset.dataframe.head(), "\n")

  # Pre-process the dataset for the training phase
  dataset.pre_process()

  if verbose:
    print("[INFO] information of the dataset pre-processed: ")
    print("\tColumns: ", dataset.dataframe_processed.columns)
    print("\tHead: \n", dataset.dataframe_processed.head(), "\n")

  # We create the weak learner with the parameters loaded
    print("[INFO] Creating the weak learners...")

  weak_learners: list[WeakLearner] = []
  for model_name in params.weak_learner.models:
    weak_learner = WeakLearner(model_name, params.weak_learner.classifier)
    weak_learners.append(weak_learner)

    if verbose:
      weak_learner.verbose = verbose
      print("\tWeak learner created with model: ", model_name)

  if verbose:
    print("\n[INFO] Total of weak learners created: ", len(weak_learners))
    print("\n[INFO] Starting the training process on the weak learners...")

  # Train the weak learners
  for weak_learner in weak_learners:
    weak_learner.train_transformer(dataset.dataframe_processed)
    weak_learner.save()

    if verbose:
      print("\tTraining completed and saved with model: ", weak_learner.model_name)

  if verbose:
    print("\n[INFO] Training process on the weak learners finished...\n")
    print("[INFO] Creating and setting up the meta learner...\n")

  # Create the meta learner
  meta_learner = MetaLearner(params.meta_learner.classifier)
  
  for weak_learner in weak_learners:
    meta_learner.add_weak_learner(weak_learner)

  if verbose:
    print("[INFO] Meta learner created with ", len(meta_learner.weak_learners), " weak learners \n")

    # Train the meta learner
    print("[INFO] Training the meta learner...")

  meta_learner.train_model(dataset.dataframe_processed)
  meta_learner.save()

  if verbose:
    print("\n[INFO] Meta learner information: ")
    print("\tEpochs used: ", meta_learner.epochs)
    print("\tBatch size: ", meta_learner.batch_size)
    print("\tProcessed embeddings shape: ", meta_learner.embeddings_processed[0].shape)
    print()

    print("[INFO] Training process on the meta learner finished...")
    print("[INFO] Meta learner saved...")
    print("[INFO] Training process finished...")

  return meta_learner
  