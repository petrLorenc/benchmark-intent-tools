Embeddings are not included because of rights issues

### Getting embeddings

You can train your own (see https://fasttext.cc/)
```
./fasttext cbow -input ../tweets_train.txt -output ../fasttext_cbow_300 -minCount 8 -dim 300 -loss ns -neg 10 -thread 20

./fasttext sent2vec -input ../tweets_train.txt -output sent2vec_300 -minCount 8 -dim 300 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 750000
```

or https://github.com/epfml/sent2vec

### Compress embeddings
Using code in repository (intent-reco/compress_model.py) and **bin_to_vec_sent2vec.py**
    
    https://github.com/Tiriar/intent-reco
