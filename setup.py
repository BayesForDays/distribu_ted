from setuptools import setup, find_packages

setup(
    name='distribu_ted',
    version='0.0.1',
    author='Cassandra Jacobs',
    author_email='jacobs.cassandra.l@gmail.com',
    license='MIT',
    url='https://github.com/BayesForDays/distribu_ted',
    description='Short exercises for training latent word representation models',
    packages=find_packages(),
    long_description='Short exercises for training latent word representation models',
    keywords=['word representations', 'TED talks', 'machine learning', 'natural language processing'],
    classifiers=[
        'Intended Audience :: Developers',
    ],
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=0.20.2',
        'scikit-learn>=0.19.2', # not currently working w/o additional nonsense
        'nltk>=3.3',
        'plotnine',
        'gensim',
        'umap-learn'
    ]
)
