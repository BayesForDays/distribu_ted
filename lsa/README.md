## Latent Semantic Analysis


A data set of *k* observations and *n* items in a vocabulary can therefore be characterized as a *k x n* matrix, where the values within the cells of this matrix correspond to counts of a given word. This **count matrix** can be large and very **sparse** -- because typically documents do not contain every word in a vocabulary. This means that if we are interested in comparing the meanings of words or comparing two observations to determine their similarity, we may very well end up with similarity scores of 0, unable to tell that "Mary likes animals" and "She likes cats" are similar sentences.

Latent semantic analysis, or LSA, is an algorithm that allows for the comparison of texts and words that gets around the sparsity issue. Specifically, the algorithm starts with individual observations (occasionally referred to as **episodes**) as bags of words and uses these observations to learn the commonalities between all the observations. As a result, each observation maintains its vector interpretation above.

Using a dimensionality reduction technique known as singular value decomposition (SVD, but also known as principle components analysis, or PCA), we can turn the large sparse matrix of word counts and make it much smaller and dense. SVD and PCA can both be used to turn the *n* vocabulary terms into some smaller number of dimensions *d*. In natural language processing applications, *d* is often a power of 2 (e.g. 64, 128, etc.) or a multiple of 100. Which dimensionality is the optimal one for a given problem is still unclear.

Another way we can think about what SVD and PCA are doing is to (try to) think of an n-dimensional shape. SVD and PCA are trying to find the longest dimensions across that sphere. See the image (from Wikipedia) below:

| ![SVD](https://upload.wikimedia.org/wikipedia/commons/e/e9/Singular_value_decomposition.gif) |
| :--: |
| *We are trying to learn the transformation that stretched this circle.* |

The end result of SVD on a count matrix is a smaller *k x d* matrix and another matrix of size *d x n* filled no longer with counts and many 0s but real numbers (e.g. -1.117, 100.380, etc.). The vectors in the *d x n* matrix we can call _word vectors_ or _word embeddings_ as they correspond to dense verses of the *n* original words we had earlier. The *k x d* matrix we get you can think of as a "latent document" matrix that can tell you how similar every observation is to every other observation. 


### Implementing LSA in Python

LSA is thankfully fairly easy to implement in Python. The basic units you will need are `scikit-learn`, which we will use for making the count matrices and running SVD, `pandas`, which we will use to load the data set into Python, and `nltk`, which we will use to break sentences into their component words (tokenization). If you have [installed this repository already](https://github.com/BayesForDays/distribu_ted), you should already have these packages. It is possible that `nltk` does not yet have a critical component that you will need for tokenization. You may need to add this line to your code:

```python
nltk.download('punkt')
```

The basic steps are as follows (choose your own file and dimensionality):

```python
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import Truncated SVD  # good with sparse matrices
import pandas as pd

df = pd.read_csv('./path/to/file')
cv = CountVectorizer(tokenizer=word_tokenize)
svd = TruncatedSVD(n_dimensions=100)
count_matrix = cv.fit_transform(df['text_column_name'])
doc_vecs = svd.fit_transform(count_matrix)
```

However, once we are there we actually need to do a bit more work to get the word vectors from the SVD model. For that you will need to dig into the model directly. You might also want to normalize because the vectors have funny lengths and you don't want every word to be most related to the word "the."

```python
word_vecs = svd.components_.T  # n x 100 matrix
normalized_word_vecs = word_vecs / np.c_[np.sqrt((word_vecs ** 2))]
```

But once you have these vectors you can very easily calculate how similar a word is to another word! We know which row in our `word_vecs` matrix corresponds to which word because we have a vocabulary in `CountVectorizer`. Specifically, `cv.get_feature_names()` will give us an ordered list of which dimension each word sits at. Similarly, `cv.vocabulary_[word]` will give you the row number for `word`.

So, if you want to find out where `pizza` is and then compare it to all the other words, you can write:

```python
pizza_vec = normalized_word_vecs[cv.vocabulary_['pizza']]
pizza_vec.T.dot(normalized_word_vecs)
```

except when you do that you don't know which word corresponds to which row. This part takes a bit more work (just some squishing data into more interpretable objects) and is covered in the notebook at `lsa.ipynb`.

## Further exercises

You can read more about how to get word-vocabulary similarities in addition to visualizing the vectors the model learns in the notebook. Both the notebook and `lsa.py` should both work out of the box if you have installed everything correctly. Please contact me if you find a bug!
