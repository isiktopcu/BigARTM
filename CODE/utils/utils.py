import artm

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from ast import literal_eval
import matplotlib.pyplot as plt
from IPython.display import clear_output


def get_top_tokens(phi, num_tokens=25, token_min_len=1, token_max_len=6):
    top_tokens = {}
    for topic_name in phi.columns:
        tokens = phi[topic_name].sort_values(ascending=False).index
        if token_min_len is not None:
            tokens = [token for token in tokens
                      if token_max_len >= len(token[1].split('_')) >= token_min_len]
        top_tokens.update({topic_name: [token[1] for token in tokens[:num_tokens]]})
    return pd.DataFrame(top_tokens)


def get_batch_dates(filename):
    from_date, to_date = filename.split('.')[0].split('_')
    return datetime.strptime(from_date, '%Y-%m-%d'), datetime.strptime(to_date, '%Y-%m-%d')


def get_step_batches(from_date, to_date, all_batches_path='ALL_BATCHES'):
    batches_dates = {f: get_batch_dates(f) for f in os.listdir(all_batches_path)}
    step_batches = [f for f, (b_from_date, b_to_date) in batches_dates.items()
                    if from_date <= b_from_date and b_to_date <= to_date]
    return step_batches


def move_step_batches(from_date, to_date, all_batches_path='ALL_BATCHES',
                      step_batches_path='STEP_BATCHES'):
    for batch in get_step_batches(from_date, to_date, all_batches_path):
        os.rename(f'{all_batches_path}/{batch}', f'{step_batches_path}/{batch}')


def move_back_step_batches(from_date, to_date, all_batches_path='ALL_BATCHES',
                           step_batches_path='STEP_BATCHES'):
    for batch in get_step_batches(from_date, to_date, step_batches_path):
        os.rename(f'{step_batches_path}/{batch}', f'{all_batches_path}/{batch}')


def copy_step_batches(from_date, to_date, all_batches_path='ALL_BATCHES',
                      step_batches_path='STEP_BATCHES'):
    for batch in get_step_batches(from_date, to_date, all_batches_path=all_batches_path):
        shutil.copy2(f'{all_batches_path}/{batch}', step_batches_path)


def clean_step_batches(step_batches_path='STEP_BATCHES'):
    shutil.rmtree(step_batches_path)
    os.mkdir(step_batches_path)


def get_vocab_to_date(to_date, vocabs_path='ARTM_VOCABS'):
    return pd.read_csv(f'{vocabs_path}/vocab_{to_date.date()}.csv', skiprows=1)


def get_theta(model, from_date, to_date, all_batches_path='ALL_BATCHES',
              step_batches_path='STEP_BATCHES'):
    clean_step_batches(step_batches_path=step_batches_path)
    copy_step_batches(from_date, to_date, all_batches_path=all_batches_path, 
                      step_batches_path=step_batches_path)
    batch_vectorizer = artm.BatchVectorizer(data_path=step_batches_path)
    theta = model.transform(batch_vectorizer=batch_vectorizer)
    clean_step_batches(step_batches_path=step_batches_path)
    return theta


def load_phi_from_csv(csv_path):
    phi = pd.read_csv(csv_path, index_col=['Unnamed: 0'])
    phi.index = [literal_eval(i) for i in phi.index]
    return phi


def load_theta_from_csv(csv_path):
    theta = pd.read_csv(csv_path, index_col=['Unnamed: 0'])
    return theta


def get_phi_theta(from_date, to_date, timestamps, models_path='MODELS',
                  all_batches_path='ALL_BATCHES', step_batches_path='STEP_BATCHES'):
    timestamp_num = [i for i, date in enumerate(timestamps[1:]) if date == to_date][0]
    model = artm.load_artm_model(f'{models_path}/model{timestamp_num}/model')
    phi = model.get_phi()
    theta = get_theta(model, from_date, to_date, all_batches_path=all_batches_path, 
                      step_batches_path=step_batches_path)
    return phi, theta


def plot_score(model, score_name='PerplexityScore', clear=True):
    if clear:
        clear_output()
    plt.plot(range(model.num_phi_updates), model.score_tracker[score_name].value)
    plt.show()


def get_model_name(path):
    return f"model{len([f for f in os.listdir(path) if not f.startswith('.')])}"


def dump_model(model, models_path):
    model_path = os.path.join(models_path, get_model_name(models_path))
    os.mkdir(model_path)
    model.dump_artm_model(os.path.join(model_path, 'model'))


def clean_bigartm_logs(work_dir):
    for file in os.listdir(work_dir):
        if file.startswith('bigartm.') or file.startswith('core.'):
            os.remove(f'{work_dir}/{file}')


def plot_delays_dist(models_delays, max_delay=10**6, xlabel='Days', 
                     ylabel='Number of extracted trend topics'):
    def filter_deltas(deltas, max_delay=max_delay):
        return [days for days in deltas if days < max_delay]
    
    fig, ax = plt.subplots()
    heights_bins = []
    for i, model in enumerate(models_delays):
        if i == 0:
            heights_bins.append(np.histogram(filter_deltas(models_delays[model].values())))
            width = (heights_bins[0][1][1] - heights_bins[0][1][0]) / (len(models_delays) + 1)
        else:
            heights_bins.append(np.histogram(filter_deltas(models_delays[model].values()), 
                                             bins=heights_bins[0][1]))
        ax.bar(heights_bins[-1][1][:-1] + width * i, heights_bins[-1][0], 
               width=width, label=model)
    
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_delays(models_delays, xlim=24, total_trends_num=91, return_stats=False,
                units=None, figsize=(7,10), save_fig=False):
    plt.figure(figsize=figsize)
    units = 'days' if units not in ['days', 'documents', 'timestamps'] else units
    yticks = []
    for model_name, trends_delays in models_delays.items():
        delays = trends_delays.values()
        num_iterations = range(7951)
        x = [x_ / 30 for x_ in num_iterations] if units == 'days' else num_iterations
        y = [len([y for y in delays if y <= days_num]) / total_trends_num for days_num in num_iterations]
        yticks.append([round(y[i], 2) for i, x_ in enumerate(x) if x_ == xlim][0])
        plt.plot(x, y, label=model_name)

    plt.xticks([i * 3 for i in range(int(xlim / 3)+1)])
    plt.yticks(yticks)
    plt.legend()
    plt.xlim((0, xlim))
    plt.ylim((0, 1))
    if save_fig:
        root_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        plt.savefig(f'{root_path}/results.jpeg')
    else:
        plt.show()

    if return_stats:
        stats = pd.DataFrame()
        for model_name, trends_delays in models_delays.items():
            model_stats = pd.DataFrame({f'delays_{model_name}': trends_delays.values()}).describe()
            stats = pd.concat([stats, model_stats], axis=1)
        return stats
