import json
import numpy as np
import pandas as pd
import pickle
import re

from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm

tqdm.pandas()


def get_top_tokens(phi, num_reorder=25):
    top_tokens = {}
    for topic_name in phi.columns:
        top_tokens.update(
            {
                topic_name: [token for token in phi[topic_name].sort_values(ascending=False) \
                                                    .iloc[:num_reorder].index]
            }
        )
    return pd.DataFrame(top_tokens)


def reorder_topic(new_phi, old_dist, new_dist, topic):
    for new_key, key in zip(new_dist, old_dist):
        new_phi.loc[[new_key], topic] = old_dist[key]


def reorder_by_popularity(phi, top_tokens, num_reorder):
    new_phi = phi.copy()
    top_tokens_values = set([colloc for sublist in top_tokens.values for colloc in sublist])
    unique_value = dict()
    for colloc in tqdm(top_tokens_values):
        unique_value[colloc] = phi.loc[
            [colloc]
        ].T.sort_values(by=[colloc], ascending=False).diff(periods=-1).abs().fillna(0)

    for topic in tqdm(phi.columns):
        dist = dict()
        top_tokens_topic = phi[topic].sort_values(ascending=False)[:num_reorder].to_dict()
        for token in top_tokens[topic]:
            dist[token] = unique_value[token].loc[topic][0]
        dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}
        reorder_topic(new_phi, dist, top_tokens_topic, topic)
    return new_phi


def get_dictionaries(vocab_path, timestamps):
    data = dict()
    for ts in tqdm(timestamps, leave=True):
        vocab = pd.read_csv(vocab_path + f"vocab_{ts.date()}.csv", header=1)
        freq = {("@collocations", token): value for token, value in zip(vocab.token, vocab[" token_tf"])}
        data[str(ts.date())] = freq
    return data


def get_token_ts_dist(vocab_path, timestamps, top_tokens):
    dists = {}
    top_tokens = set([token for sublist in top_tokens.values for token in sublist])
    freq_dicts = get_dictionaries(vocab_path, timestamps)

    for token in tqdm(top_tokens):
        ts_dist = []
        for ts in timestamps:
            if token not in freq_dicts[str(ts.date())]:
                ts_dist.append(0)
                continue
            ts_dist.append(freq_dicts[str(ts.date())][token])
        ts_dist_diff = ts_dist.copy()
        ts_dist_diff[1:] = np.diff(ts_dist)
        for ind in range(len(ts_dist_diff)):
            if ts_dist_diff[ind] < 0:
                ts_dist_diff[ind] = 0
        ts_dist_diff = [val / sum(ts_dist_diff) for ind, val in enumerate(ts_dist_diff)]
        dists[token] = ts_dist_diff
    return dists


def get_topic_ts_dist(steps):
    dists = {}
    for ind in tqdm(range(len(steps))):
        theta = steps[ind]["theta"]
        year_dist = theta.sum(axis=1) / theta.sum(axis=1).sum()
        dists.update({str(steps[ind]["date"].date()): list(year_dist.values)})
    return dists


def reorder_by_top_years_difference(token_freq, topic_freq, phi, top_tokens, num_reorder=25):
    new_phi = phi.copy()

    topic_freq = pd.DataFrame(topic_freq).T
    topic_freq.columns = top_tokens.columns

    for topic in tqdm(topic_freq.columns):
        top_tokens_topic = phi[topic].sort_values(ascending=False)[:num_reorder].to_dict()
        topic_year = topic_freq[topic].argmax()
        year_diff = dict()
        for token in top_tokens[topic]:
            token_year = np.argmax(token_freq[token])
            year_diff[token] = abs(topic_year - token_year)

        if all(value == 0 for value in year_diff.values()):
            continue

        year_diff = {k: v for k, v in sorted(year_diff.items(), key=lambda item: item[1])}
        reorder_topic(new_phi, year_diff, top_tokens_topic, topic)

    return new_phi


def reorder_by_freq_dist(token_freq, topic_freq, phi, top_tokens, num_reorder):
    new_phi = phi.copy()
    topic_freq = pd.DataFrame(topic_freq).T
    topic_freq.columns = top_tokens.columns

    for topic in tqdm(topic_freq.columns):
        top_tokens_topic = phi[topic].sort_values(ascending=False)[:num_reorder].to_dict()
        year_diff = dict()
        topic_year = topic_freq[topic]
        for token in top_tokens[topic]:
            token_year = token_freq[token][:9]
            year_diff[token] = cosine_similarity([topic_year], [token_year])[0][0]

        year_diff = {k: v for k, v in sorted(year_diff.items(), key=lambda item: item[1])}
        #         reorder_topic(new_phi, year_diff, top_tokens_topic, topic)
        for new_key, key in zip(year_diff, top_tokens_topic):
            new_phi.loc[[new_key], topic] = top_tokens_topic[key]
    return new_phi


def reorder_by_distribution(reorder_type, steps, phi, vocab_path, timestamps, top_tokens, num_reorder):
    topic_dist = get_topic_ts_dist(steps)
    token_dist = get_token_ts_dist(vocab_path, timestamps, top_tokens)

    if reorder_type == "best_year":
        return reorder_by_top_years_difference(token_dist, topic_dist, phi, top_tokens, num_reorder)
    else:
        return reorder_by_freq_dist(token_dist, topic_dist, phi, top_tokens, num_reorder)


def reorder_phi(phi, reorder_type, num_reorder, steps, vocab_path, timestamps):
    top_tokens = get_top_tokens(phi, num_reorder)
    if reorder_type == "pop":
        reordered_phi = reorder_by_popularity(phi, top_tokens, num_reorder)
    elif reorder_type in ["best_year", "freq_dist"]:
        reordered_phi = reorder_by_distribution(reorder_type, steps, phi, vocab_path, timestamps, top_tokens,
                                                num_reorder)
    return reordered_phi
