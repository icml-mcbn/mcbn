import os

wd = os.getcwd()

# Dataset path
DATA_PATH = os.path.join(wd, 'data')

# Evaluation paths
EVALUATIONS_PATH = os.path.join(wd, 'evaluations')
HYPERPARAMS_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'hyperparameters')
TAU_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'tau')
TEST_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'test')