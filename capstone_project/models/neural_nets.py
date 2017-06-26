"""This model holds function that help to create lstms in keras."""
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def create_embedding_matrix(vec_model, embedding_dim, word_index, number_words):
    """Creates and returns an embedding matrix that can be used in Keras. The function takes a vector model
    and its length as inputs. The code has been taken from:
    https://www.kaggle.com/lystdo/lstm-with-word2vec-embedding"""
    embedding_matrix = np.zeros((number_words, embedding_dim))
    for word, i in word_index.items():
        if word in vec_model.vocab:
            embedding_matrix[i] = vec_model.word_vec(word)

    print("Null word embeddings: {}".format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))
    return embedding_matrix


def create_lstm(embedding_matrix, embedding_dim, number_words, max_sequence_length=50,  num_lstm=100,
                dropout_lstm=0.2, num_dense=[50, 20], dropout_dense=0.2):
    """ Creates an lstm in keras.The basis for this code was taken from:
    https://www.kaggle.com/lystdo/lstm-with-word2vec-embedding.
    """

    embedding_layer = Embedding(number_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=dropout_lstm, recurrent_dropout=dropout_lstm)

    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_dense)(merged)

    merged = Dense(num_dense[0], activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_dense)(merged)
    merged = Dense(num_dense[1], activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_dense)(merged)

    predictions = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=predictions)

    return model
