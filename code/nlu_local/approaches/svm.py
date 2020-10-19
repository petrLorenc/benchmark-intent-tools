import numpy as np
from scipy.special import softmax

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach
from sklearn.svm import SVC


np.random.seed(1111)  # because of Illusionist OOV random generation

def logistic_regression(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []

    QS = SentenceHandler(payload, average=True)

    data = [QS.get_vector(q) for q in X_train]

    svm = SVC(probability=True)
    svm.fit(data, np.array(y_train_processed).reshape(-1, 1))

    for idx, (true_label, testing_query) in enumerate(zip(y_test, [QS.get_vector(q) for q in X_test])):
        prediction = svm.predict_proba([testing_query])
        idx_predicted = np.argsort(prediction)[0][::-1]

        confidences_over_other = [prediction[0][x] for x in idx_predicted]
        confidences_over_other_labels = [idx2label[x] for x in idx_predicted]

        similarity_data.append({"confidence": confidences_over_other[0],
                                "confidencesOther": {label: score for label, score in zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": confidences_over_other_labels[0],
                                "queryText": X_test[idx],
                                "queryHit": "log_regression"})


    return similarity_data, str(QS.emb.oov_vectors.keys())

algorithm = ["FastTextSW", "FastText", "Sent2Vec"]


if __name__ == "__main__":
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/AskUbuntuCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/ChatbotCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/WebApplicationsCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)

    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)

    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)
    stc = SentenceEmbeddingApproach("SVM", corpus_path="../../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=logistic_regression)

