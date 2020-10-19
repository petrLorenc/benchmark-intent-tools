import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('twitter_unigram.bin') # The model can be sent2vec or cbow-c+w-ngrams
vocab = model.get_vocabulary() # Return a dictionary with words and their frequency in the corpus
uni_embs, vocab = model.get_unigram_embeddings() 

with open("twitter_unigram.vec") as f:
	for word, emb in zip(vocab, uni_embs):
		f.write(word)
		f.write(" ")
		f.write(" ".join([str(x) for x in emb]))
		f.write("\n")