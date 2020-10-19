import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import class_weight

from nlu_local.embeddings.sentence import SentenceHandler
from nlu_local.utils.service_helper import SentenceEmbeddingApproach


np.random.seed(1111)  # because of Illusionist OOV random generation
num_words = 30
from scipy.special import softmax
from itertools import islice
from nlu_local.approaches.utils import adding_features, remove_contraction

def define_model(dim_in, dim_out):
    inputs = tf.keras.Input(shape=dim_in)
    x = layers.Dense(dim_out*3, activation="sigmoid")(inputs)
    x = layers.Dense(dim_out, activation="sigmoid")(x)
    outputs = layers.Activation(activation="softmax")(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='keras_model')
    model_without_softmax = tf.keras.Model(inputs=model.inputs[0], outputs=model.layers[-2].output, name='keras_model', trainable=False)

    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])

    # model_without_softmax.compile(optimizer=tf.keras.optimizers.Nadam(),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['sparse_categorical_accuracy'])

    model.summary()
    return model, model_without_softmax



def neural_network(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []
    QS = SentenceHandler(payload, average=True)

    X_train_vectors = np.array([np.concatenate([QS.get_vector(remove_contraction(q), average=True), [adding_features(remove_contraction(q))]]) for q in X_train])

    y_train_one_hot = np.array(y_train_processed).reshape(-1, 1)
    model, _ = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True )
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_processed), y_train_processed)
    model.fit(X_train_vectors, y_train_one_hot, epochs=128, batch_size=8, callbacks=[callback])

    for idx, (true_label, testing_query) in enumerate(zip(y_test, np.array([np.concatenate([QS.get_vector(remove_contraction(q), average=True), [adding_features(remove_contraction(q))]]) for q in X_test]))):
        prediction = model.predict_on_batch(testing_query.reshape(1,-1))
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


def neural_network_plus_cosine(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []
    QS = SentenceHandler(payload, average=True)

    X_train_vectors = np.array([np.concatenate([QS.get_vector(remove_contraction(q), average=True) + adding_features(remove_contraction(q))]) for q in X_train])

    y_train_one_hot = np.array(y_train_processed).reshape(-1, 1)
    model, _ = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True )
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_processed), y_train_processed)
    model.fit(X_train_vectors, y_train_one_hot, epochs=128, batch_size=8, class_weight=class_weights, callbacks=[callback])

    for idx, (true_label, testing_query) in enumerate(zip(y_test, np.array([QS.get_vector(q, average=True) for q in X_test]))):
        prediction = model.predict_on_batch(testing_query.reshape(1,-1)).numpy()
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

def neural_network_global_intents(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []
    QS = SentenceHandler(payload, average=True)

    X_train_vectors = np.array([QS.get_vector(q, average=True) for q in X_train])

    X_train_vectors_1 = []
    y_train_vectors_1 = []
    X_train_vectors_2 = []
    y_train_vectors_2 = []

    idx2label_d1 = {}
    idx2label_d2 = {}
    for k, l in idx2label.items():

        if "_L" in l:
            idx2label_d1[len(idx2label_d1)] = l
        if "_G" in l:
            idx2label_d2[len(idx2label_d2)] = l

    idx2label_d1[len(idx2label_d1)] = "OUT"

    label2idx_d1 = {v: k for k, v in idx2label_d1.items()}
    label2idx_d2 = {v: k for k, v in idx2label_d2.items()}

    for x, y in zip(X_train_vectors, y_train):
        if y in label2idx_d1:
            X_train_vectors_1.append(x)
            y_train_vectors_1.append(np.array([label2idx_d1[y]]))

        elif y in label2idx_d2:
            X_train_vectors_2.append(x)
            y_train_vectors_2.append(np.array([label2idx_d2[y]]))

            X_train_vectors_1.append(x)
            y_train_vectors_1.append(np.array([label2idx_d1["OUT"]]))

    model_1, model_without_softmax_1 = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label_d1))
    model_2, model_without_softmax_2 = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label_d2))


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    model_1.fit(np.array(X_train_vectors_1), np.array(y_train_vectors_1).reshape(-1, 1), epochs=64, batch_size=8, callbacks=[callback])

    model_2.fit(np.array(X_train_vectors_2), np.array(y_train_vectors_2).reshape(-1, 1), epochs=64, batch_size=8, callbacks=[callback])

    for idx, (true_label, testing_query) in enumerate(zip(y_test, np.array([QS.get_vector(q, average=True) for q in X_test]))):
        prediction = model_1.predict_on_batch(testing_query.reshape(1,-1)).numpy()[0]
        idx_predicted = np.argsort(prediction)[::-1]

        if idx_predicted[0] != label2idx_d1["OUT"]:
            confidences_over_other = [prediction[x] for x in idx_predicted]
            confidences_over_other_labels = [idx2label_d1[x] for x in idx_predicted]
        else:
            prediction = model_2.predict_on_batch(testing_query.reshape(1,-1)).numpy()[0]

            idx_predicted = np.argsort(prediction)[::-1]

            confidences_over_other = [prediction[x] for x in idx_predicted]
            confidences_over_other_labels = [idx2label_d2[x] for x in idx_predicted]

        # prediction = np.append(prediction_1, prediction_2)

        # idx_predicted = np.argsort(prediction)[::-1]
        #
        # confidences_over_other = [prediction[x] for x in idx_predicted]
        # confidences_over_other_labels = [idx2label[x] for x in idx_predicted]

        similarity_data.append({"confidence": str(confidences_over_other[0]),
                                "confidencesOther": {label: str(score) for label, score in zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": confidences_over_other_labels[0],
                                "queryText": X_test[idx],
                                "queryHit": "log_regression"})


    return similarity_data, str(QS.emb.oov_vectors.keys())


import numpy as np
from scipy.spatial import distance

def sample_spherical(originals, npoints, dist,  ndim):
    points = []
    for idx, orig in enumerate(originals):
        while len(points) < (npoints * (idx + 1)):
            vec = np.random.randn(ndim)
            vec /= np.linalg.norm(vec, axis=0)
            vec += orig
            vec *= dist * (0.5 + np.random.random())
            is_valid = True
            for other_orig in originals:
                if distance.euclidean(vec, other_orig) < dist:
                    is_valid = False
            if is_valid:
                points.append(vec)

    return points

def neural_network_generate_OOD(payload, X_train, y_train, X_test, y_test):
    label2idx = {l: k for k, l in enumerate(list(set(y_train)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y_train_processed = [label2idx[x] for x in y_train]

    similarity_data = []
    QS = SentenceHandler(payload)

    X_train_vectors = np.array([QS.get_vector(q, average=True) for q in X_train])

    X_train_vectors_1 = []
    y_train_vectors_1 = []
    X_train_vectors_2 = []
    y_train_vectors_2 = []

    idx2label_d1 = {}
    idx2label_d2 = {}
    for k, l in idx2label.items():

        if "_L" in l:
            idx2label_d1[len(idx2label_d1)] = l
        if "_G" in l:
            idx2label_d2[len(idx2label_d2)] = l

    idx2label_d1[len(idx2label_d1)] = "OUT"

    label2idx_d1 = {v: k for k, v in idx2label_d1.items()}
    label2idx_d2 = {v: k for k, v in idx2label_d2.items()}

    map_for_centroids = {}

    for x, y in zip(X_train_vectors, y_train):
        if y in label2idx_d1:
            X_train_vectors_1.append(x)
            y_train_vectors_1.append(np.array([label2idx_d1[y]]))

            if label2idx_d1[y] in map_for_centroids:
                map_for_centroids[label2idx_d1[y]].append(x)
            else:
                map_for_centroids[label2idx_d1[y]] = [x]

        elif y in label2idx_d2:
            X_train_vectors_2.append(x)
            y_train_vectors_2.append(np.array([label2idx_d2[y]]))

            # if label2idx_d2[y] in map_for_centroids:
            #     map_for_centroids[label2idx_d2[y]].append(x)
            # else:
            #     map_for_centroids[label2idx_d2[y]] = [x]

    centroids = []
    for key, values in map_for_centroids.items():
        centroids.append(np.array(values).mean(axis=0))

    for point in sample_spherical(originals=centroids, npoints=100, dist=100, ndim=700):
        X_train_vectors_1.append(point)
        y_train_vectors_1.append(np.array([label2idx_d1["OUT"]]))

    model_1, model_without_softmax_1 = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label_d1))
    model_2, model_without_softmax_2 = define_model(dim_in=(len(X_train_vectors[0]),), dim_out=len(idx2label_d2))


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    model_1.fit(np.array(X_train_vectors_1), np.array(y_train_vectors_1).reshape(-1, 1), epochs=200, batch_size=2, callbacks=[callback])

    model_2.fit(np.array(X_train_vectors_2), np.array(y_train_vectors_2).reshape(-1, 1), epochs=200, batch_size=2, callbacks=[callback])

    for idx, (true_label, testing_query) in enumerate(zip(y_test, np.array([QS.get_vector(q, average=True) for q in X_test]))):
        prediction = model_1.predict_on_batch(testing_query.reshape(1,-1)).numpy()[0]
        idx_predicted = np.argsort(prediction)[::-1]

        if idx_predicted[0] != label2idx_d1["OUT"]:
            confidences_over_other = [prediction[x] for x in idx_predicted]
            confidences_over_other_labels = [idx2label_d1[x] for x in idx_predicted]
        else:
            prediction = model_2.predict_on_batch(testing_query.reshape(1,-1)).numpy()[0]

            idx_predicted = np.argsort(prediction)[::-1]

            confidences_over_other = [prediction[x] for x in idx_predicted]
            confidences_over_other_labels = [idx2label_d2[x] for x in idx_predicted]

        # prediction = np.append(prediction_1, prediction_2)

        # idx_predicted = np.argsort(prediction)[::-1]
        #
        # confidences_over_other = [prediction[x] for x in idx_predicted]
        # confidences_over_other_labels = [idx2label[x] for x in idx_predicted]

        similarity_data.append({"confidence": str(confidences_over_other[0]),
                                "confidencesOther": {label: str(score) for label, score in zip(confidences_over_other_labels, confidences_over_other)},
                                "goldenIntent": true_label,
                                "intent": confidences_over_other_labels[0],
                                "queryText": X_test[idx],
                                "queryHit": "log_regression"})


    return similarity_data, str(QS.emb.oov_vectors.keys())

if __name__ == "__main__":
    path = "../../../data/corpus/from_editor/yes-no-maybe-limit-All-corpus.json"

    for limit in range(1,25):
        stc = SentenceEmbeddingApproach("Dense3", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit-{}.json".format(limit),
                                        available_embeddings=["Sent2Vec"],
                                        can_use_tfidf=[True])
        stc.evaluate(evaluation_fn=neural_network)

    stc = SentenceEmbeddingApproach("Dense3", corpus_path="../../../data/corpus/from_editor/balanced/paper-data-limit.json",
                                    available_embeddings=["Sent2Vec"],
                                    can_use_tfidf=[True])
    stc.evaluate(evaluation_fn=neural_network)

    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path=path,
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)

    # stc = SentenceEmbeddingApproach("Dense_separated",
    #                                 corpus_path=path,
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network_global_intents)

    # stc = SentenceEmbeddingApproach("Dense_generated_ood",
    #                                 corpus_path=path,
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network_generate_OOD)

    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path="../../../data/ChatbotCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)
    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path="../../../data/WebApplicationsCorpus.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)
    #
    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path="../../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)
    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path="../../../data/corpus/from_datasets/ChatbotCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)
    # stc = SentenceEmbeddingApproach("Dense",
    #                                 corpus_path="../../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json",
    #                                 available_embeddings=["Sent2Vec"],
    #                                 can_use_tfidf=[False])
    # stc.evaluate(evaluation_fn=neural_network)

