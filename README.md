# distribu_ted
### Or, a silly pun and word embeddings tutorial

## What are word embeddings?
Word embeddings are a way to represent the "meanings" of words learned by different distributional semantics methods. Words are defined not by their conceptual features but instead by how they are talked about in large corpora. Historically these have included things like Latent Semantic Analysis (LSA), Hyperspace Analogue to Language (HAL), topic models (e.g. Latent Dirichlet Allocation, or LDA), and more recent models such as GloVe, pointwise mutual information (PMI), word2vec (skip-gram or continuous bag-of-words/CBOW), ELMo, and BeRT. These models are usually estimated on massive text corpora, such as recent snapshots of Wikipedia.

## Why might I want them?

Word embeddings have proven useful for modeling cognitive language processing tasks (Landauer & Dumais, 1997; Lund & Burgess, 1996), especially for the match between a word and its semantic context or for creating semantically appropriate stimuli in behavioral experiments. These methods have been used to approximate lexical knowledge and word meaning because resources such as WordNet (Miller, 1995) have inherent limitations, are somewhat noisy, and are necessarily incomplete. Other resources such as feature norms (McRae, Cree, Seidenberg, & McNorgan, 2005) are necessarily incomplete. Plus, words or concepts might not have features, depending on your theory. Other sources of information are generally sparse. 

For predicting both human behavior and for use in natural language processing tasks, embeddings-based methods tend to lead to improvements. Even basic methods like `word2vec` can lead to huge improvements on classification. Likewise, LSA has been used in cognitive science and psycholinguistic research since it was introduced nearly 20 years ago. But, these are all models that require training, so it is important to understand what mathematical assumptions and transformations go into it. Always keep in your back pocket some skepticism about whether these models are cognitively plausible or not.

## How can I get them?

Follow this tutorial! Each model featured here is discussed in its own folder and has an associated Jupyter notebook to go along with it. All you need is a corpus and some basic tools to get started. Follow the code below to get started.

## How do I use this repository?

1. First, download the repository (as a zip file, using `git clone`, etc.)
	* If you downloaded the repository as a zip file, unzip it (wherever you unzip your files)
	* Wherever you saved this unzipped folder or cloned the repo into, you will want to `cd` into. i.e.:

	```
	cd path/to/distribu_ted
	```
	
2. Once you have cd'd into that folder, you will want to install all the basic dependencies. For that, you will need to use `pip` from within the `distribu_ted` folder all by itself. We will do a local install that will take the `setup.py` folder in this repository and include all the necessary dependencies. These dependencies include `UMAP`, `numpy`, `pandas`, `gensim`, `scikit-learn`, `nltk`, and `plotnine`. 

	```
	pip install -e .
	```
	
3. If you have also installed `jupyter`, then the notebooks (the `.ipynb` files) are all easily opened. Simply call `jupyter notebook` from within the main directory and navigate to the notebook you would like to use.
4. If not, then there are two python scripts of interest in the folders `./word2vec/word2vec.py` and `./lsa/lsa.py`. These files will go through all the necessary steps for learning embeddings using the TED talk corpus in the top level directory (`./ted_en-20160408_full.txt`).
	* To use these scripts, `cd` into the subdirectory (e.g. `./lsa/` or `./word2vec/`). 
	* If you are already very comfortable learning and training your own embeddings, you can work with the slightly more complex project in the `./tags/` folder, which visualizes the tags used in different TED talks in a 2-D space using the pre-trained embeddings that we will learn using the skip-gram model ([Mikolov et al., 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)).

## References

### Models
* **LSA**: Landauer, T. K., & Dumais, S. T. (1997). A solution to Plato's problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge. _Psychological Review, 104_, 211-240.
* **HAL**: Lund, K., & Burgess, C. (1996). Producing high-dimensional semantic spaces from lexical co-occurrence. _Behavior Research Methods, Instruments, & Computers, 28_, 203-208.
* **LDA**: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. _Journal of Machine Learning Research, 3_, 993-1022.
* **word2vec (skip-gram and CBOW)**: Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In _Advances in Neural Information Processing Systems_ (pp. 3111-3119).
* **GloVe**: Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ (pp. 1532-1543).
* **PMI**: Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. _Computational linguistics, 16_, 22-29.
* **ELMo**: Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep Contextualized Word Representations. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ (Vol. 1, pp. 2227-2237).
* **BERT**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint_. arXiv:1810.04805.

### Resources
* Miller, G. A. (1995). WordNet: a lexical database for English. _Communications of the ACM, 38_, 39-41.
* McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. _Behavior Research Methods, 37_, 547-559.
