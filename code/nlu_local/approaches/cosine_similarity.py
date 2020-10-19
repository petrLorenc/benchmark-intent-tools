import numpy as np
from scipy.special import softmax

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach


np.random.seed(1111)  # because of Illusionist OOV random generation

boosted = False
boosted_words = ['enjoying', 'quiet', 'launched', 'again', 'opened', 'absolutely', 'none', 'sleeping', "doesn't", 'love', 'times', 'still', 'off', 'for', 'agreed', 'start', 'wanna', 'lovely', 'less', 'farewell', 'spoken', 'what', 'that', 'never', 'volume', 'exactly', 'indifferent', 'nah', 'turned', 'not', 'false', 'saying', 'idea', 'care', 'this', 'few', 'other', 'goodbye', 'negative', "don't", 'right', 'quietly', 'be', 'want', 'yep', 'me', 'now', 'moment', 'sure', 'once', 'of', 'open', 'talked', "can't", 'yup', 'down', 'keep', 'quieter', 'yes', 'ok', 'course', 'you', 'matter', 'great', 'no', 'cannot', 'leave', 'repeat', 'alright', 'stop', 'sucks', 'boring', 'bet', 'nice', 'which', 'hear', 'higher', 'yeah', 'quit', 'understand', 'correct', 'more', 'play', 'have', 'started', 'may', 'close', 'games', 'stopped', 'game', 'wait', 'it', 'really', 'up', 'speaking', 'silently', 'agree', 'talking', 'positive', 'come', 'bye', 'memory', 'might', 'end', 'ahead', 'all', 'loud', 'know', 'nope', 'okey', 'hate', 'yea', 'question', 'maybe', 'how', 'okay', 'shut', 'pause', 'go', 'true', 'possibly', 'exit', 'definitely', 'get', 'probably', 'one', 'please', 'too', 'louder', 'enough', 'session', 'different', 'later', 'turn', 'remember', 'many', 'talk', 'speak', 'neither', 'played', 'low', 'available', 'say', 'asleep']


# confidences, ids, based_on_similarity = self.similarity_fn(X, vec, n)
def similarity_fn(X, vec, n):
    based_on_similarity = True
    sim = np.dot(X, vec)

    if n > 1:
        limit = sim.shape[0] if sim.shape[0] < n else n
        ids = np.argpartition(sim, -limit)[-limit:]
        ids = reversed(ids[np.argsort(sim[ids])])
    else:
        ids = [np.argmax(sim)]

    return sim, ids, based_on_similarity



def cosine_similarity(payload, X_train, y_train, X_test, y_test):
    similarity_data = []
    QS = SentenceHandler(payload, average=True, similarity_fn=similarity_fn, boosted_words=boosted_words)

    training_queries = [QS.get_vector(q, boosted=boosted) for q in X_train]
    testing_queries = [QS.get_vector(q, boosted=boosted) for q in X_test]

    corr = np.inner(testing_queries, training_queries)

    for idx, (true_label, testing_query) in enumerate(zip(y_test, X_test)):
        idx_closest = np.argsort(corr[idx])[::-1]
        i = 0

        predicted_label = y_train[idx_closest[i]]
        closest_sentence = X_train[idx_closest[i]]

        confidences_over_other = []
        confidences_over_other_labels = []
        for i in idx_closest:
            if y_train[i] not in confidences_over_other_labels:
                confidences_over_other_labels.append(y_train[i])
                confidences_over_other.append(corr[idx][i])

        confidences_over_other = softmax(confidences_over_other)

        similarity_data.append({"confidence" : confidences_over_other[0],
                                "confidencesOther": {label: score for label, score in zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": predicted_label,
                                "queryText": testing_query,
                                "queryHit": closest_sentence})


    return similarity_data, str(QS.emb.oov_vectors.keys())

algorithm = ["FastTextSW", "FastText", "Sent2Vec"]


if __name__ == "__main__":
    # for limit in range(1,25):
    #     stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-{}.json".format(limit),
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    #     stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine_cz_Sent2vec", corpus_path="../../../data/corpus/from_editor/cz/intro-cz-processed.json",
    #                             available_embeddings=["Sent2Vec"],
    #                             can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity, lang="cs")
    #
    # stc = SentenceEmbeddingApproach("Cosine_cz_FastTextSW", corpus_path="../../../data/corpus/from_editor/cz/intro-cz-processed.json",
    #                             available_embeddings=["FastTextSW"],
    #                             can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity, lang="cs")

    files = [
        # "../../../data/AskUbuntuCorpus.json",
        # "../../../data/ChatbotCorpus.json",
        # "../../../data/WebApplicationsCorpus.json",
        #
        # "../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
        # "../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
        # "../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
        #
        # "../../../data/corpus/from_editor/balanced/paper-data-limit.json",
        # "../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
        # "../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
        "../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json"
    ]

    for file in files:
        stc = SentenceEmbeddingApproach("Cosine_300_fasttext_tfidf_embeddings", corpus_path=file,
                                        available_embeddings=["FastTextSW"],
                                        can_use_tfidf=[True])
        stc.evaluate(evaluation_fn=cosine_similarity)

        stc = SentenceEmbeddingApproach("Cosine_300_sent2vec_tfidf_embeddings", corpus_path=file,
                                        available_embeddings=["Sent2Vec"],
                                        can_use_tfidf=[True])
        stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW_tfidf", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[True])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/AskUbuntuCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/AskUbuntuCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)


    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    #
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine_fasttextSW", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
    #                                 available_embeddings=["FastTextSW"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)

    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)
    # stc = SentenceEmbeddingApproach("Cosine", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=cosine_similarity)



