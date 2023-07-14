import pandas as pd
from datetime import datetime
from tqdm import tqdm
from .trends_topics import get_trends_from_murkup, get_topics_from_topic_model
from .trend_topics_matching import trends_topics_match
from .utils import get_phi_theta


def get_docs_dates(collection_path='COLLOCATIONS.csv'):
    collocations = pd.read_csv(collection_path, parse_dates=['conf_pub_date', 'arxiv_pub_date'],
                               date_parser=lambda d: datetime.strptime(d, '%Y-%m-%d') if not pd.isna(d) else None)
    collocations['min_date'] = collocations.apply(lambda row: min(row.conf_pub_date, row.arxiv_pub_date), axis=1)
    docs_dates = {collocations.iloc[i]['paper_id']: collocations.iloc[i]['min_date']
                  for i in range(len(collocations))}
    return docs_dates


def get_steps(timestamps, models_path='MODELS', all_batches_path='ALL_BATCHES', 
              step_batches_path='STEP_BATCHES'):
    zero_date = timestamps[0]
    steps = []
    for i, to_date in enumerate(tqdm(timestamps[1:])):
        steps.append({'date': to_date})
        steps[i]['phi'], steps[i]['theta'] = get_phi_theta(zero_date, to_date, timestamps, models_path,
                                                           all_batches_path, step_batches_path)
    return steps


def get_steps_scores(steps, k_d=50, k_w=50, k_s=10, collection_path='COLLOCATIONS.csv', 
                     markup_path='trends_markup.csv'):
    docs_dates = get_docs_dates(collection_path)
    zero_date = min(docs_dates.values())
    
    trends = get_trends_from_murkup(markup_path)

    for step in tqdm(steps):
        interval_trends = [trend.from_interval(docs_dates, zero_date, step['date']) for trend in trends]
        interval_trends = [trend for trend in interval_trends if len(trend.key_docs) > 0]

        phi = step['phi']
        theta = step['theta']
        step_topics = get_topics_from_topic_model(phi, theta)

        trends_statuses, trends_best_topics, best_topics_scores = trends_topics_match(interval_trends, step_topics, 
                                                                                      k_d=k_d, k_w=k_w, k_s=k_s)
        step['scores'] = {interval_trends[j].name: {'is_matched': trends_statuses[j],
                                                    'best_topic': trends_best_topics[j],
                                                    'best_topic_scores': best_topics_scores[j]}
                          for j in range(len(interval_trends))}
    return steps


def get_trend_detection_date(trend, steps, threshold_d=None, threshold_w=None, use_s=True):
    def w_criterion(score): return score > threshold_w if threshold_w is not None else True
    def d_criterion(score): return score > threshold_d if threshold_d is not None else True
    def s_criterion(score): return score if use_s else True

    true_trend_dates = [step['date'] for step in steps if trend.name in step['scores']
                        and w_criterion(step['scores'][trend.name]['best_topic_scores']['M_W'])
                        and d_criterion(step['scores'][trend.name]['best_topic_scores']['M_D'])
                        and s_criterion(step['scores'][trend.name]['best_topic_scores']['M_S'])]
    return min(true_trend_dates) if true_trend_dates else None


def get_trends_detection_dates(trends, steps, threshold_d=None, threshold_w=None, use_s=False):
    return {trend: get_trend_detection_date(trend, steps, threshold_d, threshold_w, use_s) 
            for trend in trends}


def get_trends_detection_delays(steps, collection_path='COLLOCATIONS.csv', 
                                markup_path='trends_markup.csv', threshold_d=None, 
                                threshold_w=None, use_s=False, units=None, timestamps=None):
    units = 'days' if units not in ['days', 'documents', 'timestamps'] else units
    trends = get_trends_from_murkup(markup_path)

    docs_dates = get_docs_dates(collection_path)
    trends_dates_true = {trend.name: trend.get_docs_interval(docs_dates)[0] for trend in trends}

    trends_detection_dates = get_trends_detection_dates(trends, steps,
                                                        threshold_d, threshold_w, use_s)
    if units == 'days':
        return {trend: (trends_detection_dates[trend] - trends_dates_true[trend.name]).days
                for trend in trends_detection_dates if trends_detection_dates[trend] is not None}
    elif units == 'documents':
        delays_in_docs = {}
        for trend in trends_detection_dates:
            if trends_detection_dates[trend] is not None:
                docs_increment = [paper_id for paper_id, date in docs_dates.items()
                                  if trends_dates_true[trend.name] <= date <= trends_detection_dates[trend]]
                delays_in_docs[trend] = len(docs_increment)
        return delays_in_docs
    else:
        delays_in_timestamps = {}
        for trend in trends_detection_dates:
            if trends_detection_dates[trend] is not None:
                timestamps_increment = [ts for ts in timestamps
                                        if trends_dates_true[trend.name] < ts <= trends_detection_dates[trend]]
                delays_in_timestamps[trend] = len(timestamps_increment)
        return delays_in_timestamps
