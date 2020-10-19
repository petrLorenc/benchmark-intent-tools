# -*- coding: utf-8 -*-
"""
Code is removed because of privacy issues - code for working with quantized embeddings can be found https://github.com/Tiriar/intent-reco
"""

class SentenceHandler:
    """
    Question-answering by semantic query-matching.
    :param data: QA data (see above)
    """

    def __init__(self, data=None, average=False, similarity_fn=None, boosted_words=None):
        pass


    def tfidf_vector(self, sentence):
        """
        Transform sentence into vector form using TF-IDF weights.
        :param sentence: input sentence
        :return: weighted sentence vector
        """
        vec = None
        return vec

    def boosted_vector(self, sentence):
        """
        Transform sentence into vector form using TF-IDF weights - boost some keywords.
        :param sentence: input sentence
        :return: weighted sentence vector
        """

        vec = None
        return vec

    def tokenize(self, sentence):
        """
        Tokenize sentence into words.
        Used for TF-IDF word tokenizer that requires list of tokens.
        :param sentence: input sentence
        :return: list of word tokens
        """

        tokens = None
        return tokens

    def get_vector(self, query, average=True, boosted=False):
        if boosted:
            return self.boosted_vector(query)

        return self.emb.vector(query, average=average) if self.TFIDF is None else self.tfidf_vector(query)
