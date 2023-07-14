import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


def recall_at_k(key_items, items_rank, k, is_terms=False):
    if not is_terms:
        return len(set(items_rank[:k]) & key_items) / len(key_items)

    found_keys = 0
    key_items_ = [set(term.split('_')) for term in key_items]
    for key_item in key_items_:
        items_rank_ = [item_rank for length in range(len(key_item), 6) for item_rank in items_rank[length][:k]]
        for item_rank in items_rank_:
            if key_item <= item_rank:
                found_keys += 1
                break
    return found_keys / len(key_items)


def get_similarities(trends, topics, k_d=25, k_w=25, k_s=10):
    terms_as_set = [[term, set(term.split('_'))] for term in topics[0].terms_rank]
    terms_as_set = pd.DataFrame([[term, term_as_set, len(term_as_set)] for term, term_as_set in terms_as_set],
                                columns=['term', 'term_as_set', 'len'])
    topics_lens_dict = {}
    for topic in topics:
        items_rank = pd.DataFrame({'term': topic.terms_rank}).merge(terms_as_set, how='left', on='term')
        lens_dict = {length: items_rank[items_rank.len == length].term_as_set.values for length in range(1, 6)}
        topics_lens_dict[topic.name] = lens_dict

    M_D, M_W, M_S = np.zeros((3, len(trends), len(topics)))
    for i, trend in enumerate(trends):
        for j, topic in enumerate(topics):
            M_D[i][j] = recall_at_k(trend.key_docs, topic.docs_rank, k_d)
            M_W[i][j] = recall_at_k(trend.key_terms, topics_lens_dict[topic.name], k_w, is_terms=True)
            M_S[i][j] = recall_at_k(trend.name_synonyms, topics_lens_dict[topic.name], k_s, is_terms=True)
    return M_D, M_W, M_S > 0


def trends_topics_match(trends, topics, threshold_d=0.5, threshold_w=0.5,
                        k_d=25, k_w=25, k_s=10, get_matrices=False):
    M_D, M_W, M_S = get_similarities(trends, topics, k_d=k_d, k_w=k_w, k_s=k_s)
    trends_statuses = ((M_D >= threshold_d) & (M_W >= threshold_w)).max(axis=1)
    trends_best_topics = (M_D + M_W + M_S).argmax(axis=1)
    best_topics_scores = [{'M_D': M_D[i][best_topic], 'M_W': M_W[i][best_topic], 'M_S': M_S[i][best_topic]}
                          for i, best_topic in enumerate(trends_best_topics)]
    if get_matrices:
        return trends_statuses, trends_best_topics, best_topics_scores, M_D, M_W, M_S
    else:
        return trends_statuses, trends_best_topics, best_topics_scores

# TESTS
# from trends_topics import Topic, Trend
# from datetime import datetime


# test_trends = [Trend(['doc1', 'doc2', 'doc3'], ['1', '2', '3'], ['trend1', 't1'], 'trend1'),
#                Trend(['doc4', 'doc5', 'doc6'], ['8', '5', '6'], ['trend2', 't2'], 'trend2'),
#                Trend(['doc4', 'doc5', 'doc6'], ['4', '5', '6'], ['trend3', 't3'], 'trend3')]

# test_topics = [Topic(['doc1', 'doc4', 'doc3', 'doc2', 'doc5', 'doc6'],
#                      ['1_2_3', '7', '2', '3', '6', 't1', 't2', 't3', '5'], 'topic1'),
#                Topic(['doc4', 'doc5', 'doc6', 'doc1', 'doc2', 'doc3'],
#                      ['4', '5', '1', '6', '2', '3', '1_2_3', 't2', 't1', 't3', '7'], 'topic2'),
#                Topic(['doc3', 'doc1', 'doc6', 'doc4', 'doc2', 'doc5'],
#                      ['1', '2', '5', '1_2_3', '7', '3', 't2', 't1', 't3'], 'topic3')]

# test_docs_dates = {
#     'doc1': datetime(2013, 1, 1),
#     'doc2': datetime(2013, 4, 1),
#     'doc3': datetime(2013, 3, 1),
#     'doc4': datetime(2013, 4, 1),
#     'doc5': datetime(2013, 5, 1),
#     'doc6': datetime(2013, 6, 1),
# }

# M_D, M_W, M_S = get_similarities(test_trends, test_topics, k_w=3, k_d=3, k_s=5)
# trends_statuses, trends_best_topics, \
#     best_topics_scores, M_D, M_W, M_S = trends_topics_match(test_trends, test_topics, k_w=3, 
#                                                             k_d=3, k_s=5, get_matrices=True)
# trends_statuses, trends_best_topics, best_topics_scores, M_D, M_W, M_S