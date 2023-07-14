import pandas as pd


class Trend:
    def __init__(self, key_docs, key_terms, name_synonyms, name=None):
        self.name = name
        self.key_docs = set(key_docs)
        self.key_terms = set(key_terms)
        self.name_synonyms = set(name_synonyms)

    def from_interval(self, docs_dates, from_date, to_date):
        interval_key_docs = [doc for doc in self.key_docs if from_date <= docs_dates[doc] <= to_date]
        return Trend(interval_key_docs, self.key_terms, self.name_synonyms, self.name)

    def get_docs_interval(self, docs_dates):
        dates = [docs_dates[doc] for doc in self.key_docs]
        return min(dates), max(dates)

    def __repr__(self):
        return f'Trend {self.name} with {len(self.key_docs)} key_docs and {len(self.key_terms)} key_terms. ' \
               f'Trend name synonyms: {self.name_synonyms}.'


class Topic:
    def __init__(self, docs_rank, terms_rank, name=None):
        self.name = name
        self.docs_rank = docs_rank
        self.terms_rank = terms_rank

    def from_interval(self, docs_dates, from_date, to_date):
        interval_docs_rank = [doc for doc in self.docs_rank if from_date <= docs_dates[doc] <= to_date]
        return Topic(interval_docs_rank, self.terms_rank, self.name)

    def __repr__(self):
        return f'Topic {self.name} on D: |D| = {len(self.docs_rank)} and W: |W| = {len(self.terms_rank)}'


def get_trends_from_murkup(markup_path, min_key_docs_num=5, min_key_terms_num=5, collection=None):
    def filter_papers_ids(papers_ids):
        return [paper_id for paper_id in papers_ids if paper_id in collection.paper_id.values]

    markup_df = pd.read_csv(markup_path, sep=';')
    markup_df.paper_ids = markup_df.paper_ids.apply(lambda paper_ids: set(paper_ids.split(',')))
    markup_df.key_collocations = markup_df.key_collocations.apply(lambda key_cols: set(key_cols.split(',')))
    markup_df.trend_name_synonyms = markup_df.trend_name_synonyms.apply(lambda name_synonyms: set(name_synonyms.split(',')))

    if collection:
        markup_df['paper_ids'] = markup_df['paper_ids'].apply(filter_papers_ids)

    markup_df = markup_df[(markup_df.paper_ids.apply(len) >= min_key_docs_num) &
                          (markup_df.key_collocations.apply(len) >= min_key_terms_num)]
    return [Trend(key_docs, key_terms, name_synonyms, name)
            for name, key_docs, key_terms, name_synonyms in markup_df.to_records(index=False)]


def get_topics_from_topic_model(phi, theta):
    terms_ranks = {topic_name: [term for _, term in phi[topic_name].sort_values(ascending=False).index.tolist()]
                   for topic_name in phi.columns}
    docs_ranks = {topic_name: theta.T[topic_name].sort_values(ascending=False).index.unique().tolist()
                  for topic_name in theta.T.columns}
    return [Topic(docs_ranks[topic_name], terms_ranks[topic_name], topic_name) for topic_name in terms_ranks]
