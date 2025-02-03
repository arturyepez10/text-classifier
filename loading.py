from models.learner.weak_learner import WeakLearner
from models.learner.meta_learner import MetaLearner

def load_model(filename: str, verbose: bool = False):

  if verbose:
    print("[INFO] Loading Meta-Learner model from the path provided")

  meta_learner = MetaLearner.load(filename)

  if verbose:
    print("[INFO] Meta-Learner model loaded successfully...")
    print("\n[INFO] Meta-Learner model information: ")
    print("\tWeak learners: ", len(meta_learner.weak_learners))
    for weak_learner in meta_learner.weak_learners:
      print("\t\tModel: ", weak_learner.model_name)
      print("\t\tEmbeddings shape: ", weak_learner.embeddings.shape)
      print()

    print("\tEmbeddings shape: ", meta_learner.embeddings_processed.shape)
    print("\tPredictions per epoch shape: ", meta_learner.predictions_per_epoch.shape)

    print("\n[INFO] Meta learner loaded...")
    print("[INFO] Loading process finished...")

  return meta_learner