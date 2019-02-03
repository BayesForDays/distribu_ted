## word2vec (skip-gram)



### Implementing word2vec in Python

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
