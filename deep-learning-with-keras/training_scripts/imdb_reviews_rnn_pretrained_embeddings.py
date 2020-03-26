'''
Train a simple RNN on IMDB reviews dataset.
'''


import os
from argparse import ArgumentParser

from keras.layers import Dense, Dropout, Flatten, RNN, Embedding
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_reviews(directory):
    '''
    Read reviews from text files and return a list of reviews with their
    labels.
    '''
    data = []
    labels = []
    print(f'Getting {directory.split("/")[-1]} reviews ...')
    for label in ['neg', 'pos']:
        sub_dir = os.path.join(directory, label)

        for fname in os.listdir(sub_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(sub_dir, fname), 'r') as f:
                    data.append(f.read())
            if label == 'pos':
                labels.append(1)
            else:
                labels.append(0)
    labels = np.array(labels)

    return data, labels


def get_word_embeddings(args):
    embeddings_index = {}
    print('Getting pretrained words embeddings ...')
    with open(args.embeddings_dir, 'r') as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            values = np.array(data[1:],dtype='float32')
            embeddings_index[word] = values

    embeddings_dim = len(values)

    return embeddings_index, embeddings_dim


def get_embeddings_matrix(embeddings_index, word_index, embeddings_dim, args):
    # Get embeddings matrix
    embeddings_matrix = np.zeros((args.max_words, embeddings_dim))

    for word, i in word_index.items():
        if i < args.max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

    return embeddings_matrix


def preprocess_text(train_text, test_text, args):
    '''Tokenize text and pad the resulted sequences.'''
    print('Preprocessing reviews ...')
    # Tokenize text based on training text
    tokenizer = Tokenizer(args.max_words)
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index

    # Preprocess training text
    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_sequences = pad_sequences(train_sequences, maxlen=args.max_length)

    # Preprocess training text
    test_sequences = tokenizer.texts_to_sequences(test_text)
    test_sequences = pad_sequences(test_sequences, maxlen=args.max_length)

    return train_sequences, test_sequences, word_index


def train(X_train, y_train, embeddings_matrix, args):
    '''
    Train a simple model using Embedding layer followed by fully connected
    layers.
    '''

    model = models.Sequential()
    model.add(Embedding(args.max_words,
                        embeddings_matrix.shape[1],
                        input_length=args.max_length))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Freeze embedding layer
    model.layers[0].set_weights([embeddings_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print('Started training ...')
    model.fit(X_train,
              y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=0.2,
              verbose=args.verbose)
    print('Training is done.')

    return model


def test(model, X_test, y_test):
    '''Evaluate model on test text.'''
    _, acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy : {acc:.2%}.')


def main():
    parser = ArgumentParser(
        description='Train a simple RNN on IMDB reviews dataset')
    parser.add_argument('--data_dir', type=str, metavar='',
                        help='Directory containing train and test subdir')
    parser.add_argument('--embeddings_dir', type=str, metavar='',
                        help='Directory containing pretrained embeddings')
    parser.add_argument('--batch_size', type=int, default=32, metavar='',
                        help='Batch size to be used when training the CNN')
    parser.add_argument('--epochs', type=int, default=20, metavar='',
                        help='Number of full training cycles')
    parser.add_argument('--max_words', type=int, default=1000, metavar='',
                        help='`max_words` most frequent words in the text')
    parser.add_argument('--max_length', type=int, default=100, metavar='',
                        help='length of each sequence to pad/truncate')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='Verbosity mode')
    args = parser.parse_args()

    # Define train and test dir
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

    # Get training and test reviews
    train_text, train_labels = get_reviews(train_dir)
    test_text, test_labels = get_reviews(test_dir)

    # Preprocess reviews
    train_seq, test_seq, word_index = preprocess_text(
        train_text, test_text, args)

    # Shuffle training data
    idxs = np.random.permutation(len(train_labels))
    train_seq = train_seq[idxs]
    train_labels = train_labels[idxs]

    # Get embeddings matrixs
    embeddings_index, embeddings_dim = get_word_embeddings(args)
    embeddings_matrix = get_embeddings_matrix(
        embeddings_index, word_index, embeddings_dim, args)

    # Train the model
    model = train(train_seq, train_labels, embeddings_matrix, args)
    test(model, test_seq, test_labels)


if __name__ == '__main__':
    main()
