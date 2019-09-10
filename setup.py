import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='word2vec-keras',
    version='0.1',
    author="Paolo Ripamonti",
    author_email="paolo.ripamonti93@gmail.com",
    description="Word2Vec Keras Text Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javatechy/dokr",
    packages=setuptools.find_packages(),
    keywords=['keras', 'word2vec', 'deep learning', 'machine learning'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gensim",
        "tensorflow",
        "keras",
        "numpy",
        "sklearn",
        "tqdm"
    ]
)
