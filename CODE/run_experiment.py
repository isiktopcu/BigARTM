import argparse
import pickle

from models import *

models = {
    'plsa': update_model_plsa,
    'lda': update_model_lda,
    'artm_decor': update_model_decor,
    'artm_decor_sp_theta': update_model_decor_sp_theta
}


def create_work_dirs(kwargs):
    def create_work_dir(dir_path, clean=False):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        elif clean:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)

    create_work_dir(kwargs['batches_path'])
    create_work_dir(kwargs['step_batches_path'], clean=True)
    create_work_dir(kwargs['vocabs_path'])
    create_work_dir(kwargs['temp_path'], clean=True)
    create_work_dir(kwargs['models_path'], clean=True)


def run_experiment(kwargs):
    create_work_dirs(kwargs)
    with open(kwargs['timestamps_path'], 'rb') as f:
        timestamps = pickle.load(f)

    fitted_models = []
    model = None
    from_date = timestamps[0]
    for i, to_date in enumerate(timestamps[1:], 1):
        print('='*52 + f' STEP {i} ' + '='*52)
        if kwargs['increment_mode']:
            from_date = timestamps[i-1]
        model, _, __, ___ = reshape_model(timestamps[i-1], to_date, kwargs, model)
        model = models[kwargs['model_type']](from_date, to_date, kwargs, model)
        dump_model(model, kwargs['models_path'])
        fitted_models.append(model.clone())
    clean_bigartm_logs('.')


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_type", help="topic model type", default='artm_decor_sp_theta')
parser.add_argument("--models_path", help="path to fitted models", default=f'{root_path}/models')
parser.add_argument("--batches_path", help="path to batches", default=f'{root_path}/DATA/batches')
parser.add_argument("--step_batches_path", help="path to batches of current step",
                    default=f'{root_path}/step_batches')
parser.add_argument("--vocabs_path", help="path to vocabs for timestamps", default=f'{root_path}/DATA/vocabs')
parser.add_argument("--timestamps_path", help="path to timestamps", default=f'{root_path}/DATA/timestamps.pickle')
parser.add_argument("--temp_path", help="path to temp files", default=f'{root_path}/temp')
parser.add_argument("--increment_mode",
                    help="0 – use all docs until current timestamp. 1 – use docs between 2 last timestamps", default=1)
parser.add_argument("--early_stop_eps", help="early stop value of perplexity score", default=0.05)
parser.add_argument("--patience", help="number of steps with perplexity less then early stop value", default=3)
parser.add_argument("--max_collection_passes", help="maximum number of collection passes", default=24)
parser.add_argument("--sparsity_phi_threshold", help="threshold beyond which decorrelator phi (regularizer) turns off",
                    default=0.9)
parser.add_argument("--sparsity_theta_threshold", help="threshold beyond which sparse theta regularizer turns off",
                    default=0.9)
parser.add_argument("--start_num_topics", help="start number of topics", default=200)

run_experiment(vars(parser.parse_args()))
