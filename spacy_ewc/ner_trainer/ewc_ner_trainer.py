from thinc.api import set_dropout_rate


def train_ner_with_ewc(ner, examples, ewc, *, drop=0.0, sgd=None, losses=None):
    """
    This function is based on `spacy.pipeline.transition_parser.Parser`, with the addition of EWC penalty
    application. The EWC penalty step has been integrated to adjust gradients during training to retain
    previous knowledge.
    """
    if losses is None:
        losses = {}
    losses.setdefault(ner.name, 0.0)
    set_dropout_rate(ner.model, drop)
    states, golds, _ = ner.moves.init_gold_batch(examples)
    model, backprop_tok2vec = ner.model.begin_update([eg.x for eg in examples])
    states_golds = list(zip(states, golds))
    while states_golds:
        states, golds = zip(*states_golds)
        scores, backprop = model.begin_update(states)
        d_scores = ner.get_batch_loss(states, golds, scores, losses)

        backprop(d_scores)
        ner.transition_states(states, scores)
        states_golds = [(s, g) for (s, g) in zip(states, golds) if not s.is_final()]

    backprop_tok2vec(golds)
    # Execute EWC penalty calculation
    ewc.apply_ewc_penalty_to_gradients()
    if sgd not in (None, False):
        ner.finish_update(sgd)
    del backprop
    del backprop_tok2vec
    model.clear_memory()
    del model
    return losses


def train_nlp_with_ewc(nlp, examples, ewc, *, drop=0.0, sgd=None, losses=None):
    """
    Wrapper function to train the `nlp` pipeline with EWC applied to NER gradients.
    Based on `spacy.pipeline.transition_parser.Parser`, modified to include EWC penalty.
    """
    return train_ner_with_ewc(
        ner=nlp.get_pipe("ner"),
        examples=examples,
        ewc=ewc,
        drop=drop,
        sgd=sgd,
        losses=losses,
    )
