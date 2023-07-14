from utils.utils import *


def reshape_model(from_date, to_date, h_params, model=None, new_terms_to_new_topics=False, num_processors=-1):
    def get_new_topics_num(from_date_, to_date_, old_topics_num_):
        vocab_from_date = get_vocab_to_date(from_date_, h_params['vocabs_path'])
        vocab_to_date = get_vocab_to_date(to_date_, h_params['vocabs_path'])
        new_terms = vocab_to_date[~vocab_to_date.token.isin(vocab_from_date.token)]
        return int(len(new_terms) / len(vocab_from_date) * old_topics_num_)

    vocab = artm.Dictionary()
    vocab.load_text(f"{h_params['vocabs_path']}/vocab_{to_date.date()}.csv")
    if model is None:
        model = artm.ARTM(num_topics=h_params['start_num_topics'], dictionary=vocab, num_processors=num_processors,
                          show_progress_bars=True, class_ids={'@collocations': 1}, theta_columns_naming='title')
        model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore', class_id='@collocations'))
        model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        model.scores.add(artm.PerplexityScore(name='PerplexityScore'))
        return model, None, None, None
    else:
        old_topics_num = len(model.topic_names)
        new_topics_num = get_new_topics_num(from_date, to_date, old_topics_num)
        new_topics_names = [f'{to_date.date()}_{i}' for i in range(new_topics_num)]

        reshaped_model = model.clone()
        reshaped_model.reshape_tokens(vocab)

        model_phi = model.get_phi(model_name='nwt')
        reshaped_model_phi = reshaped_model.get_phi(model_name='nwt')

        old_tokens_idx = {token[1]: i for i, token in enumerate(model_phi.index)}
        reshaped_tokens_idx = {token[1]: i for i, token in enumerate(reshaped_model_phi.index)}
        init_dict = {reshaped_tokens_idx[token]: old_tokens_idx[token]
                     for token in reshaped_tokens_idx if token in old_tokens_idx}
        new_tokens_idx = {token: i for token, i in reshaped_tokens_idx.items()
                          if token not in old_tokens_idx}

        _, nwt_matrix = model.master.attach_model('nwt')
        _, reshaped_nwt_matrix = reshaped_model.master.attach_model('nwt')
        for new_token_idx, old_token_idx in init_dict.items():
            reshaped_nwt_matrix[new_token_idx] = nwt_matrix[old_token_idx]
        reshaped_model.master.normalize_model(reshaped_model._model_pwt, reshaped_model._model_nwt)

        reshaped_model.reshape_topics(model.topic_names + new_topics_names)
        pwt_model, pwt_matrix = reshaped_model.master.attach_model('pwt')
        new_cols = np.random.rand(pwt_matrix.shape[0], new_topics_num)
        if new_terms_to_new_topics:
            zero_idx = [i for i in range(new_cols.shape[0]) if i not in new_tokens_idx.values()]
            new_cols[zero_idx] = 0

        new_cols = np.array([col / sum(col) for col in new_cols.T]).T
        pwt_matrix[:, old_topics_num: old_topics_num + new_topics_num] = new_cols

        return reshaped_model, nwt_matrix, pwt_matrix, reshaped_nwt_matrix


def update_model_decor(from_date, to_date, h_params, model, clear=False):
    copy_step_batches(from_date, to_date, all_batches_path=h_params['batches_path'],
                      step_batches_path=h_params['step_batches_path'])
    batch_vectorizer = artm.BatchVectorizer(data_path=h_params['step_batches_path'])

    not_raise = 0
    for i in range(h_params['max_collection_passes']):
        tau = 0
        if 'SparsityPhiScore' in model.score_tracker:
            if model.score_tracker['SparsityPhiScore'].value[-1] < h_params['sparsity_phi_threshold']:
                tau = 0.2
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi',
                                                               class_ids=['@collocations'],
                                                               tau=tau, gamma=0), overwrite=True)
        model.fit_offline(batch_vectorizer, num_collection_passes=1)

        if i > 0:
            perp_score_before, perp_score_after = model.score_tracker['PerplexityScore'].value[-2:]
            if (perp_score_before - perp_score_after) / perp_score_before < h_params['early_stop_eps']:
                not_raise += 1
                if not_raise > h_params['patience']:
                    plot_score(model, clear=clear)
                    break
            else:
                not_raise = 0
                plot_score(model, clear=clear)

    clean_step_batches(step_batches_path=h_params['step_batches_path'])
    return model


def update_model_decor_sp_theta(from_date, to_date, h_params, model,
                                clear=False, plot_perplexity=False, plot_sparsity=False):
    copy_step_batches(from_date, to_date, all_batches_path=h_params['batches_path'],
                      step_batches_path=h_params['step_batches_path'])
    batch_vectorizer = artm.BatchVectorizer(data_path=h_params['step_batches_path'])

    not_raise = 0
    for i in range(h_params['max_collection_passes']):
        tau_decor = 0
        if 'SparsityPhiScore' in model.score_tracker:
            if model.score_tracker['SparsityPhiScore'].value[-1] < h_params['sparsity_phi_threshold']:
                tau_decor = 0.2
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi',
                                                               class_ids=['@collocations'],
                                                               tau=tau_decor, gamma=0), overwrite=True)
        tau_sp_theta = 0
        if 'SparsityThetaScore' in model.score_tracker:
            if model.score_tracker['SparsityThetaScore'].value[-1] < h_params['sparsity_theta_threshold']:
                tau_sp_theta = -1
        model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmSpTheta',
                                                                 tau=tau_sp_theta), overwrite=True)

        model.fit_offline(batch_vectorizer, num_collection_passes=1)

        if plot_sparsity:
            plt.plot(range(model.num_phi_updates),
                     model.score_tracker['SparsityPhiScore'].value)

            plt.plot(range(model.num_phi_updates),
                     model.score_tracker['SparsityThetaScore'].value, 'b--')
            plt.grid(True)
            plt.show()

        if i > 0:
            perp_score_before, perp_score_after = model.score_tracker['PerplexityScore'].value[-2:]
            if (perp_score_before - perp_score_after) / perp_score_before < h_params['early_stop_eps']:
                not_raise += 1
                if not_raise > h_params['patience']:
                    if plot_perplexity:
                        plot_score(model, clear=clear)
                    break
            else:
                not_raise = 0
                if plot_perplexity:
                    plot_score(model, clear=clear)

    clean_step_batches(step_batches_path=h_params['step_batches_path'])
    return model


def update_model_plsa(from_date, to_date, h_params, model, clear=False, plot_perplexity=False):
    copy_step_batches(from_date, to_date, all_batches_path=h_params['batches_path'],
                      step_batches_path=h_params['step_batches_path'])
    batch_vectorizer = artm.BatchVectorizer(data_path=h_params['step_batches_path'])

    not_raise = 0
    for i in range(h_params['max_collection_passes']):
        model.fit_offline(batch_vectorizer, num_collection_passes=1)

        if i > 0:
            perp_score_before, perp_score_after = model.score_tracker['PerplexityScore'].value[-2:]
            if (perp_score_before - perp_score_after) / perp_score_before < h_params['early_stop_eps']:
                not_raise += 1
                if not_raise > h_params['patience']:
                    if plot_perplexity:
                        plot_score(model, clear=clear)
                    break
            else:
                not_raise = 0
                if plot_perplexity:
                    plot_score(model, clear=clear)

    clean_step_batches(step_batches_path=h_params['step_batches_path'])
    return model


def update_model_lda(from_date, to_date, h_params, model, alpha=None, betta=None,
                     plot_perplexity=False, clear=False):
    copy_step_batches(from_date, to_date, all_batches_path=h_params['batches_path'],
                      step_batches_path=h_params['step_batches_path'])
    batch_vectorizer = artm.BatchVectorizer(data_path=h_params['step_batches_path'])
    alpha = 1 / model.num_topics if alpha is None else alpha
    betta = 1 / model.num_topics if betta is None else alpha

    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='RegularizerDirichletTheta',
                                                             tau=alpha), overwrite=True)
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='RegularizerDirichletPhi',
                                                           class_ids=['@collocations'],
                                                           tau=betta), overwrite=True)

    not_raise = 0
    for i in range(h_params['max_collection_passes']):
        model.fit_offline(batch_vectorizer, num_collection_passes=1)

        if i > 0:
            perp_score_before, perp_score_after = model.score_tracker['PerplexityScore'].value[-2:]
            if (perp_score_before - perp_score_after) / perp_score_before < h_params['early_stop_eps']:
                not_raise += 1
                if not_raise > h_params['patience']:
                    if plot_perplexity:
                        plot_score(model, clear=clear)
                    break
            else:
                not_raise = 0
                if plot_perplexity:
                    plot_score(model, clear=clear)

    clean_step_batches(step_batches_path=h_params['step_batches_path'])
    return model
