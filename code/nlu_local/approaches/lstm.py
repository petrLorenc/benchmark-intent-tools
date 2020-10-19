import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import class_weight

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach


np.random.seed(1111)  # because of Illusionist OOV random generation
num_words = 30


def define_model(dim_in, dim_out):
    inputs = tf.keras.Input(shape=dim_in)
    x = layers.Masking()(inputs)
    x = layers.Bidirectional(layers.LSTM(300, recurrent_dropout=0.7, dropout=0.7))(x)
    outputs = layers.Dense(dim_out, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='nn_model')

    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model


def padding_vector(X):
    result = np.zeros((num_words, X.shape[1]))
    result[:min(X.shape[0], num_words), : X.shape[1]] = X[:min(X.shape[0], num_words), : X.shape[1]]
    return result


def neural_network(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []
    QS = SentenceHandler(payload)

    X_train_vectors = np.array([padding_vector(QS.get_vector(q, average=False)) for q in X_train])
    y_train_one_hot = np.array(y_train_processed).reshape(-1, 1)
    model = define_model(dim_in=(len(X_train_vectors[0]), len(X_train_vectors[0][0])), dim_out=len(idx2label))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_processed), y_train_processed)
    model.fit(X_train_vectors, y_train_one_hot, epochs=256, batch_size=8, class_weight=class_weights, callbacks=[callback])

    for idx, (true_label, testing_query) in enumerate(zip(y_test, np.array([padding_vector(QS.get_vector(q, average=False)) for q in X_test]))):
        prediction = model.predict_on_batch(testing_query.reshape(1, num_words, -1)).numpy()
        idx_predicted = np.argsort(prediction)[0][::-1]

        confidences_over_other = [prediction[0][x] for x in idx_predicted]
        confidences_over_other_labels = [idx2label[x] for x in idx_predicted]

        similarity_data.append({"confidence": str(confidences_over_other[0]),
                                "confidencesOther": {label: str(score) for label, score in zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": confidences_over_other_labels[0],
                                "queryText": X_test[idx],
                                "queryHit": "log_regression"})


    return similarity_data, str(QS.emb.oov_vectors.keys())

algorithm = ["FastTextSW", "FastText", "Sent2Vec"]


if __name__ == "__main__":
    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data/AskUbuntuCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)
    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data//ChatbotCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)
    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data/WebApplicationsCorpus.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)

    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)
    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)
    stc = SentenceEmbeddingApproach("LSTM",
                                    corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[False])
    stc.evaluate(evaluation_fn=neural_network)

