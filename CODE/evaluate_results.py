import argparse
import pickle

from utils.evaluation import *
from utils.utils import *


def evaluate_results(kwargs):
    with open(kwargs['timestamps_path'], 'rb') as f:
        timestamps = pickle.load(f)

    if kwargs['load_steps']:
        with open(f"{os.path.basename(kwargs['models_path'])}.pickle", 'rb') as f:
            steps = pickle.load(f)
    else:
        steps_num = len(os.listdir(kwargs['models_path']))
        steps = get_steps(timestamps[:steps_num], kwargs['models_path'], kwargs['batches_path'],
                          kwargs['step_batches_path'])
        steps = get_steps_scores(steps, collection_path=kwargs['collection_path'], markup_path=kwargs['markup_path'])

        with open(f"{os.path.basename(kwargs['models_path'])}.pickle", 'wb') as f:
            pickle.dump(steps, f)

    models_delays = {}
    for config in tqdm([
        {'threshold_d': None, 'threshold_w': 0.3, 'use_s': False},
        {'threshold_d': None, 'threshold_w': 0.3, 'use_s': True},
        {'threshold_d': 0.1, 'threshold_w': 0.3, 'use_s': False},
        {'threshold_d': 0.1, 'threshold_w': 0.3, 'use_s': True},
        {'threshold_d': 0.1, 'threshold_w': None, 'use_s': False},
        {'threshold_d': 0.1, 'threshold_w': None, 'use_s': True}
    ]):
        delays = get_trends_detection_delays(steps,
                                             collection_path=kwargs['collection_path'],
                                             markup_path=kwargs['markup_path'],
                                             threshold_d=config['threshold_d'],
                                             threshold_w=config['threshold_w'],
                                             use_s=config['use_s'],
                                             units=kwargs['units'],
                                             timestamps=timestamps)
        grid_point = f"{kwargs['model_type']}_D{config['threshold_d']}_W{config['threshold_w']}_S{config['use_s']}"
        models_delays[grid_point] = delays

    plot_delays(models_delays, xlim=30, units=kwargs['units'], total_trends_num=91,
                figsize=(5, 7), save_fig=kwargs['save_fig'])
    clean_bigartm_logs('.')


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_type", help="topic model type", default='?')
parser.add_argument("--models_path", help="path to fitted models", default=f'{root_path}/models')
parser.add_argument("--batches_path", help="path to batches", default=f'{root_path}/DATA/batches')
parser.add_argument("--step_batches_path", help="path to batches of current step",
                    default=f'{root_path}/step_batches')
parser.add_argument("--collection_path", help="path to collection", default=f'{root_path}/DATA/AITD/collection.csv')
parser.add_argument("--markup_path", help="path to markup", default=f'{root_path}/DATA/AITD/trends_markup.csv')
parser.add_argument("--timestamps_path", help="path to timestamps", default=f'{root_path}/DATA/timestamps.pickle')
parser.add_argument("--units", help="units of delays: timestamps, days or documents", default='days')
parser.add_argument("--load_steps", help="0 – create step from models. 1 – load steps from pickle.", default=0)
parser.add_argument("--save_fig", help="0 – show results plot in window. 1 – save results plot into root dir as jpeg.",
                    default=1)

evaluate_results(vars(parser.parse_args()))
