# Word2Vec-Keras Text Classifier
Word2Vec-Keras is a simple Word2Vec and LSTM wrapper for text classification.

It combines Gensim Word2Vec model with Keras neural network trhough an Emnedding layer as input.
The Neural Network works with LSTM layer

## How install
```git
pip3 install git+https://github.com/paoloripamonti/word2vec-keras
```

## Usage

```python
from word2vec_keras import Word2VecKeras

model = Word2VecKeras()

model.train(x_train, y_train)
```

Train Word2Vec and Keras models

Train parameters:
- **x_train**: list of raw sentences, no text cleaning will be perfomed
- **y_train**: list of labels
- **w2v_size**: (Default: 300) Word2Vec - Dimensionality of the word vectors
- **w2v_window**: (Default: 5) Word2Vec - Maximum distance between the current and predicted word within a sentence.
- **w2v_min_count**: (Default: 1) Word2Vec - Ignores all words with total frequency lower than this.
- **w2v_epochs**: (Default: 100) Word2Vec - Number of iterations (epochs) over the corpus.
- **k_max_sequence_len**: (Default: 500) Keras - Maximum length of all sequences
- **k_batch_size**:(Default: 128) Keras - Number of samples per gradient update
- **k_epochs**:(Default: 32) Keras - Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
- **k_lstm_neurons**: (Default: 128) Keras - LSTM neurons per layer
- **k_hidden_layer_neurons**: (Default: \[128, 64, 32]) Keras - Number of Dense layers after LSTM layer.
- **verbose**: (Default: 1) Keras- 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

### Evaluate

```python

model.evaluate(x_test, y_test)
```

Evaluate model

Evaluate parameters:
- **x_test**: list of raw sentences, no text cleaning will be perfomed
- **y_test**: list of labels

Evaluate result:
- Return a dictionary with ACCURAY, CLASSIFICATION_REPORT and CONFUSION_MATRIX


### Predict

```python
model.predict('lorem ipsum dolor sit amet consectetur adipiscing elit...', threshold=0.6)
```

Make prediction of give text

Predict parameters:
- **x_text**: Raw text, no text cleaning will be perfomed
- **threshold**: (Default: 0.0) Cut-off threshold, if confidence il less than given value return __OTHER__ as label

Predict result:
- Return a dictionary with LABEL, CONFIDENCE and ELAPSED_TIME, i.e. {label: LABEL, confidence: CONFIDENCE, elapsed_time: TIME}

### Save & load model

```python
model.save('/path/model.tar.gz')
```

Save model as compressed **tar.gz** file that contains several utility pickles, keras model and Word2Vec model

```python
model = Word2VecKeras()

model.load('/path/model.tar.gz')
```

Load model from saved tar.gz file

## Example

```python
from sklearn.datasets import fetch_20newsgroups
from word2vec_keras import Word2VecKeras
from pprint import pprint

# fetch the dataset using scikit-learn
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

train_b = fetch_20newsgroups(subset='train',
                             categories=categories, shuffle=True, random_state=42)
test_b = fetch_20newsgroups(subset='test',
                            categories=categories, shuffle=True, random_state=42)

print('size of training set: %s' % (len(train_b['data'])))
print('size of validation set: %s' % (len(test_b['data'])))
print('classes: %s' % (train_b.target_names))

x_train = train_b.data
y_train = [train_b.target_names[idx] for idx in train_b.target]
x_test = test_b.data
y_test = [train_b.target_names[idx] for idx in test_b.target]

model = Word2VecKeras()
model.train(x_train, y_train)

pprint(model.evaluate(x_test, y_test))

model.save('./model.tar.gz')

```
