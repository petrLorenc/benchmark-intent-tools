import numpy as np
from scipy.special import softmax

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach
from sklearn.linear_model import LogisticRegression
from nlu_local.approaches.utils import adding_features

np.random.seed(1111)  # because of Illusionist OOV random generation

boosted = False
boosted_words = ['enjoying', 'quiet', 'launched', 'again', 'opened', 'absolutely', 'none', 'sleeping', "doesn't",
                 'love', 'times', 'still', 'off', 'for', 'agreed', 'start', 'wanna', 'lovely', 'less', 'farewell',
                 'spoken', 'what', 'that', 'never', 'volume', 'exactly', 'indifferent', 'nah', 'turned', 'not', 'false',
                 'saying', 'idea', 'care', 'this', 'few', 'other', 'goodbye', 'negative', "don't", 'right', 'quietly',
                 'be', 'want', 'yep', 'me', 'now', 'moment', 'sure', 'once', 'of', 'open', 'talked', "can't", 'yup',
                 'down', 'keep', 'quieter', 'yes', 'ok', 'course', 'you', 'matter', 'great', 'no', 'cannot', 'leave',
                 'repeat', 'alright', 'stop', 'sucks', 'boring', 'bet', 'nice', 'which', 'hear', 'higher', 'yeah',
                 'quit', 'understand', 'correct', 'more', 'play', 'have', 'started', 'may', 'close', 'games', 'stopped',
                 'game', 'wait', 'it', 'really', 'up', 'speaking', 'silently', 'agree', 'talking', 'positive', 'come',
                 'bye', 'memory', 'might', 'end', 'ahead', 'all', 'loud', 'know', 'nope', 'okey', 'hate', 'yea',
                 'question', 'maybe', 'how', 'okay', 'shut', 'pause', 'go', 'true', 'possibly', 'exit', 'definitely',
                 'get', 'probably', 'one', 'please', 'too', 'louder', 'enough', 'session', 'different', 'later', 'turn',
                 'remember', 'many', 'talk', 'speak', 'neither', 'played', 'low', 'available', 'say', 'asleep']


def logistic_regression(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []

    if boosted:
        QS = SentenceHandler(payload, average=True, boosted_words=boosted_words)
    else:
        QS = SentenceHandler(payload, average=True)

    data = [np.append(QS.get_vector(q, boosted=boosted), adding_features(q), axis=0) for q in X_train]

    log_regression = LogisticRegression(C=1000, max_iter=10000, multi_class="ovr", random_state=42)
    log_regression.fit(data, np.array(y_train_processed).reshape(-1, 1))

    for idx, (true_label, testing_query) in enumerate(zip(y_test, [np.append(QS.get_vector(q, boosted=boosted), adding_features(q), axis=0) for q in X_test])):
        prediction = log_regression.predict_proba([testing_query])
        idx_predicted = np.argsort(prediction)[0][::-1]

        confidences_over_other = [prediction[0][x] for x in idx_predicted]
        confidences_over_other_labels = [idx2label[x] for x in idx_predicted]

        similarity_data.append({"confidence": confidences_over_other[0],
                                "confidencesOther": {label: score for label, score in
                                                     zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": confidences_over_other_labels[0],
                                "queryText": X_test[idx],
                                "queryHit": "log_regression"})

    return similarity_data, str(QS.emb.oov_vectors.keys())


algorithm = ["FastTextSW", "FastText", "Sent2Vec"]

if __name__ == "__main__":
    files = [
        "../../../data/AskUbuntuCorpus.json",
        "../../../data/ChatbotCorpus.json",
        "../../../data/WebApplicationsCorpus.json",

        "../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
        "../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
        "../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
        #
        "../../../data/corpus/from_editor/balanced/paper-data-limit.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json"
    ]

    for file in files:
        stc = SentenceEmbeddingApproach("LogisticRegression_700_sent2vec_contraction", corpus_path=file,
                                        available_embeddings=["Sent2Vec"],
                                        can_use_tfidf=[False])
        stc.evaluate(evaluation_fn=logistic_regression)

    # for limit in range(1,25):
    #     stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-{}.json".format(limit),
    #                                     available_embeddings=["Sent2Vec"],
    #                                     can_use_tfidf=[True])
    #     stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW_tfidf",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/AskUbuntuCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/AskUbuntuCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("LogisticRegression", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
