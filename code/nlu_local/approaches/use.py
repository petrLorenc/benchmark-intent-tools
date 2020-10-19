import numpy as np
from scipy.special import softmax
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach

np.random.seed(1111)

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model_USE = hub.load(module_url)
print("module %s loaded" % module_url)


def embed(input):
    return model_USE(input)


def logistic_regression(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []

    log_regression = LogisticRegression(C=1000, max_iter=10000)
    log_regression.fit(embed(X_train).numpy().reshape(-1, 512), np.array(y_train_processed).reshape(-1, 1))

    for idx, (true_label, testing_query) in enumerate(zip(y_test, embed(X_test).numpy())):
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

    return similarity_data, str("USE")


algorithm = ["FastTextSW", "FastText", "Sent2Vec"]

if __name__ == "__main__":

    stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-H-F.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTRAIN-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-ONLYTEST-H-F-D.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/AskUbuntuCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    #
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg",
    #                                 corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusTypos50.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)
    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)


    # stc = SentenceEmbeddingApproach("USE_Log_reg", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusReplace.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=logistic_regression)

