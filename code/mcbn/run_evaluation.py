import os
import argparse
from mcbn.utils.helper import get_logger, get_setup, dump_yaml

NOTEBOOKS = ['01. Dataset splitter.ipynb',
             '02. lambda, batch size, dropout grid search.ipynb',
             '03. Get best grid search results.ipynb',
             '04. Tau optimization.ipynb',
             '05. Get best tau results.ipynb',
             '06. Test set evaluation.ipynb',
             '07. Collect-test-results.ipynb']

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
logger = get_logger()

def configure_setup(dataset_name, split_seed):
    logger.info('Evaluating dataset: {} split seed: {}'.format(dataset_name, split_seed))
    setup = get_setup()
    setup['datasets'] = [dataset_name]
    setup['split_seed'] = split_seed
    dump_yaml(setup, os.getcwd(), 'setup.yml')

def run_evaluation():
    for nb_name in NOTEBOOKS:
        script_name = nb_name.replace('.ipynb', '.py')
        os.system("{}/runnb.sh '{}' '{}'".format(CURR_DIR, nb_name, script_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run entire evaluation procedure for a certain dataset and fold')
    parser.add_argument('dataset_name', help='Dataset to evaluate')
    parser.add_argument('split_seed', type=int, help='Split of dataset to evaluate')
    args = parser.parse_args()

    configure_setup(args.dataset_name, args.split_seed)
    run_evaluation()
    logger.info("ALL DONE")