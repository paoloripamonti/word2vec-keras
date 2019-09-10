# Word2vec
import gensim

# Scikit learn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Keras
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Utility
import numpy as np
import tarfile
import pickle
import tempfile
import os
import time
import logging
import multiprocessing
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Word2VecKeras(object):
    """
    Combine word2vec with keras model in order to build strong text classifier
    """

    def __init__(self):
        """
        Initialize empty classifier
        """
        self.w2v_size = None
        self.w2v_window = None
        self.w2v_min_count = None
        self.w2v_epochs = None
        self.label_encoder = None
        self.num_classes = None
        self.tokenizer = None
        self.k_max_sequence_len = None
        self.k_batch_size = None
        self.k_epochs = None
        self.k_lstm_neurons = None
        self.k_hidden_layer_neurons = None
        self.w2v_model = None
        self.k_model = None

    def train(self, x_train, y_train, w2v_size=300, w2v_window=5, w2v_min_count=1,
              w2v_epochs=100, k_max_sequence_len=500, k_batch_size=128, k_epochs=32, k_lstm_neurons=128,
              k_hidden_layer_neurons=(128, 64, 32), verbose=1):
        """
        Train new Word2Vec & Keras model
        :param x_train: list of sentence
        :param y_train: list of labels
        :param w2v_size: Word2Vec vector size
        :param w2v_window: Word2Vec windows size
        :param w2v_min_count: Word2Vec min word count
        :param w2v_epochs: Word2Vec epochs number
        :param k_max_sequence_len: Max sequence length
        :param k_batch_size: Keras training batch size
        :param k_epochs: Keras epochs number
        :param k_lstm_neurons: neurons number for Keras LSTM layer
        :param k_hidden_layer_neurons: array of keras hidden layers
        :param verbose: Verbosity
        """
        # Set variables
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count
        self.w2v_epochs = w2v_epochs
        self.k_max_sequence_len = k_max_sequence_len
        self.k_batch_size = k_batch_size
        self.k_epochs = k_epochs
        self.k_lstm_neurons = k_lstm_neurons
        self.k_hidden_layer_neurons = k_hidden_layer_neurons

        # split text in tokens
        x_train = [text.split() for text in x_train]

        logging.info("Build & train Word2Vec model")
        self.w2v_model = gensim.models.Word2Vec(min_count=self.w2v_min_count, window=self.w2v_window,
                                                size=self.w2v_size,
                                                workers=multiprocessing.cpu_count())
        self.w2v_model.build_vocab(x_train)
        self.w2v_model.train(x_train, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_epochs)
        w2v_words = list(self.w2v_model.wv.vocab)
        logging.info("Vocabulary size: %i" % len(w2v_words))
        logging.info("Word2Vec trained")

        logging.info("Fit LabelEncoder")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train)
        self.num_classes = len(self.label_encoder.classes_)
        y_train = utils.to_categorical(y_train, self.num_classes)

        logging.info("Fit Tokenizer")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(x_train)
        x_train = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences(x_train),
                                                             maxlen=self.k_max_sequence_len)
        num_words = len(self.tokenizer.word_index) + 1
        logging.info("Number of unique words: %i" % num_words)

        logging.info("Create Embedding matrix")
        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.w2v_size))
        for word, idx in word_index.items():
            if word in w2v_words:
                embedding_vector = self.w2v_model.wv.get_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = self.w2v_model.wv[word]
        logging.info("Embedding matrix: %s" % str(embedding_matrix.shape))

        logging.info("Build Keras model")
        logging.info('x_train shape: %s' % str(x_train.shape))
        logging.info('y_train shape: %s' % str(y_train.shape))

        self.k_model = Sequential()
        self.k_model.add(Embedding(vocab_size,
                                   self.w2v_size,
                                   weights=[embedding_matrix],
                                   input_length=self.k_max_sequence_len,
                                   trainable=False))
        self.k_model.add(LSTM(self.k_lstm_neurons, dropout=0.5, recurrent_dropout=0.2))
        for hidden_layer in self.k_hidden_layer_neurons:
            self.k_model.add(Dense(hidden_layer, activation='relu'))
            self.k_model.add(Dropout(0.2))
        if self.num_classes > 1:
            self.k_model.add(Dense(self.num_classes, activation='softmax'))
        else:
            self.k_model.add(Dense(self.num_classes, activation='sigmoid'))

        self.k_model.compile(loss='categorical_crossentropy' if self.num_classes > 1 else 'binary_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
        logging.info(self.k_model.summary())

        # Callbacks
        early_stopping = EarlyStopping(monitor='acc', patience=6, verbose=0, mode='max')
        rop = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=6, verbose=1, epsilon=1e-4, mode='max')
        callbacks = [early_stopping, rop]

        logging.info("Fit Keras model")
        self.k_model.fit(x_train, y_train,
                         batch_size=self.k_batch_size,
                         epochs=self.k_epochs,
                         callbacks=callbacks,
                         verbose=verbose)

        logging.info("Done")

    def predict(self, text, threshold=.0):
        """
        Predict raw text label
        :param text: raw text
        :param threshold: cut-off threshold, if confidence il less than given value return __OTHER__ as label
        :return: {label: LABEL, confidence: CONFIDENCE, elapsed_time: TIME}
        """
        if not self.k_model or not self.w2v_model:
            raise RuntimeError("Model not in memory, please load it train new model")
        start_at = time.time()
        x_test = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([text]),
                                                            maxlen=self.k_max_sequence_len)
        # Predict
        confidences = self.k_model.predict(x_test)[0]
        # Get mex prediction
        idx = np.argmax(confidences)
        elapsed_time = time.time() - start_at
        if float(confidences[idx]) > threshold:
            return {"label": self.label_encoder.classes_[idx], "confidence": float(confidences[idx]),
                    "elapsed_time": elapsed_time}
        return {"label": "__OTHER__", "confidence": float(confidences[idx]), "elapsed_time": elapsed_time}

    def evaluate(self, x_test, y_test):
        """
        Evaluate Model with several KPI
        :param x_test: Text to test
        :param y_test: labels for text
        :return: dictionary with KPIs
        """
        result = {}
        results = []
        # Predict
        for text in tqdm(x_test):
            results.append(self.predict(text, threshold=0.0))

        y_pred_1d = []
        y_pred_scores = []

        for i in range(0, len(results)):
            y_pred_scores.append(results[i]["confidence"])
            y_pred_1d.append(results[i]["label"])

        y_pred_bin = []
        for i in range(0, len(results)):
            y_pred_bin.append(1 if y_pred_1d[i] == y_test[i] else 0)

        # Classification report
        result["CLASSIFICATION_REPORT"] = classification_report(y_test, y_pred_1d, output_dict=True)

        # Confusion matrix
        result["CONFUSION_MATRIX"] = confusion_matrix(y_test, y_pred_1d)

        # Accuracy
        result["ACCURACY"] = accuracy_score(y_test, y_pred_1d)

        return result

    def save(self, path="word2vec_keras.tar.gz"):
        """
        Save all models in pickles file
        :param path: path to save
        """
        tokenizer_path = os.path.join(tempfile.gettempdir(), "tokenizer.pkl")
        label_encoder_path = os.path.join(tempfile.gettempdir(), "label_encoder.pkl")
        params_path = os.path.join(tempfile.gettempdir(), "params.pkl")
        keras_path = os.path.join(tempfile.gettempdir(), "model.h5")
        w2v_path = os.path.join(tempfile.gettempdir(), "model.w2v")

        # Dump pickle
        pickle.dump(self.tokenizer, open(tokenizer_path, "wb"))
        pickle.dump(self.label_encoder, open(label_encoder_path, "wb"))
        pickle.dump(self.__attributes__(), open(params_path, "wb"))
        pickle.dump(self.w2v_model, open(w2v_path, "wb"))
        self.k_model.save(keras_path)
        # self.w2v_model.save(w2v_path)

        # Create Tar file
        tar = tarfile.open(path, "w:gz")
        for name in [tokenizer_path, label_encoder_path, params_path, keras_path, w2v_path]:
            tar.add(name, arcname=os.path.basename(name))
        tar.close()

        # Remove temp file
        for name in [tokenizer_path, label_encoder_path, params_path, keras_path, w2v_path]:
            os.remove(name)

    def load(self, path):
        """
        Load all attributes from path
        :param path: tar.gz dump
        """
        # Open tarfile
        tar = tarfile.open(mode="r:gz", fileobj=open(path, "rb"))

        # Iterate over every member
        for filename in tar.getnames():
            if filename == "model.h5":
                self.k_model = load_model(tar.extractfile(filename))
            if filename == "model.w2v":
                self.w2v_model = pickle.loads(tar.extractfile(filename).read())
            if filename == "tokenizer.pkl":
                self.tokenizer = pickle.loads(tar.extractfile(filename).read())
            if filename == "label_encoder.pkl":
                self.label_encoder = pickle.loads(tar.extractfile(filename).read())
            if filename == "params.pkl":
                params = pickle.loads(tar.extractfile(filename).read())
                for k, v in params.items():
                    self.__setattr__(k, v)

    def __attributes__(self):
        """
        Attributes to dump
        :return: dictionary
        """
        return {
            "w2v_size": self.w2v_size,
            "w2v_window": self.w2v_window,
            "w2v_min_count": self.w2v_min_count,
            "w2v_epochs": self.w2v_epochs,
            "num_classes": self.num_classes,
            "k_max_sequence_len": self.k_max_sequence_len,
            "k_batch_size": self.k_batch_size,
            "k_epochs": self.k_epochs,
            "k_lstm_neurons": self.k_lstm_neurons,
            "k_hidden_layer_neurons": self.k_hidden_layer_neurons
        }
