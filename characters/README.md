## Characters also have distributional properties

It is possible to learn character embeddings -- representations of letters in written language -- that capture the contexts in which we use them. For example, "t" and "p" might be similar partly because they can both occur before the letter "h". 

The notebook in this directory is a very simple introduction to an unconstrained problem -- what can we do to learn the best representations of characters? What information matters most? Here are some potential thoughts:

* Does it matter whether a letter occurs at the beginning of a word or not? (Test the effect of adding "w\_s" and "w\_e" characters)
* Does it matter whether the input is lower or upper case? (Test what happens if the input is all provided in lower case)
* What letters do we expect to see behave similarly to each other? (Extract the most similar letters to each letter)
* Do we want to learn digraphs and other more complex combinations (e.g. "-ng", "-ough")? (Test adding the Phrases module from gensim to learn digraphs).

	```python
	from gensim.models.phrases import Phrases, Phraser
	from gensim.models.word2vec import Word2Vec
	
	sentences = [['This', 'is', 'a', 'sentence'],
	['Here', 'is', 'another', one']]
	
	phrases = Phrases(sentences, min_count=1, threshold=1)
	chunks = Phraser(phrases)
	sentences_chunked = [chunks[x] for x in sentences]
	w2v = Word2Vec(sentences_chunked)
	```