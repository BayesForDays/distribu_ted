## word2vec (skip-gram)

### What is word2vec?

word2vec is a general family of models that are generally thought to "predict" words. The general task is not to find the main topics that characterize a set of documents, which is the objective of LSA. Instead, there are two general ways of training the model, known typically as skip-gram and continuous bag-of-words (CBOW). word2vec is well-established to provide better fits to behavioral data (Mandera, Keuleers, & Brysbaert, 2016) and many more natural language processing tasks than LSA.

#### What is continuous bag-of-words?

Continuous bag-of-words, or CBOW (see-bow) is an algorithm that takes a given chunk of words and takes one word out to be predicted. The remaining words predict the final word. Because the model only has to predict one word at a time, learning good word representations is a less difficult problem than learning them in the skip-gram model. CBOW is thought to be less good at low-frequency words because the model has little incentive to learn much about rarer words. On the other hand, it tends to train more quickly than skip-gram.

#### What is skip-gram?

Skip-gram is a model that uses a single word to predict some set of words to the left and another set of words to the right. It is called skip-gram because it is the "skipped" word that is being used to predict the surrounding context. Skip-gram is thought to do better with rarer words than CBOW, perhaps because of a trick used to speed up training known as negative sampling.

Each word is allocated its own "slot" in the predictions. So, if the model is to predict five words to the left and five words to the right of a given word, then the model predicts up to 10 times the size of the vocabulary. This obviously gets very hairy because for large vocabularies this could easily entail predicting nearly 100,000 outcomes for reasonably-sized corpora. Negative sampling is one solution to this. Rather than ask the model to predict only words in the context, adding "negative words" speeds model training. It also seems to improve the model's ability to discriminate between the real context and fake contexts. 

## Implementing word2vec in Python

LSA is thankfully fairly easy to implement in Python. We only really need `pandas`, which we will use to load the data set into Python, and `nltk`, which we will use to break sentences into their component words (tokenization). We will use the `gensim` package to create word vectors. If you have [installed this repository already](https://github.com/BayesForDays/distribu_ted), you should already have these packages. It is possible that `nltk` does not yet have a critical component that you will need for tokenization. You may need to add this line to your code:

```
nltk.download('punkt')
```

The basic steps are as follows (choose your own file and dimensionality):

```
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

algorithm = 1 # skip-gram
context_window_size = 5

df = pd.read_csv('./path/to/file')
w2v = Word2Vec([word_tokenize(x) for x in df['text_column']], 
				sg = 1, window_size = context_window_size)
```

Unlike LSA, once you have trained the word2vec model, you can very easily calculate how similar a word is to another word through the word2vec API. Specifically, you can call `w2v.wv.most_similar(word)` and it will give you a list of the top most similar words along with their cosine similarity.

Getting the vectors out is slightly trickier but still fairly straightforward:

```
vecs = w2v.wv.get_vector(word)
```

It is also much easier to save this model and reload it for other applications:

```
w2v.wv.save('./path/to/file.model')
```


## Further exercises

You can read more about how to get word-vocabulary similarities in addition to visualizing the vectors the model learns in the notebook. Both the notebook and `word2vec.py` should both work out of the box if you have installed everything correctly. Please contact me if you find a bug!

## References
* Mandera, P., Keuleers, E., & Brysbaert, M. (2017). Explaining human performance in psycholinguistic tasks with models of semantic similarity based on prediction and counting: A review and empirical validation. _Journal of Memory and Language, 92_, 57-78.

