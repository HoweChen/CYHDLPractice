from keras.datasets import imdb
import numpy as np
from keras import models, layers


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences)), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    word_index = imdb.get_word_index()

    reverse_word_index = dict([(value, key) for key, value in word_index.items()])
    decoded_review = "".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("flat32")

    model = models.Sequential()
    model.add(layers=layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layer=layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layer=layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
