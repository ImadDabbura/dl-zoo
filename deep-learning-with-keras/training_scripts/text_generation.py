'''
Use NN with LSTM to generate random text using provided text file. The model
is called char-level language model that tries to predict next character given
all previous characters.
'''
import os
import sys
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers


def read_file(file_name):
    '''Read the text file in memory.'''
    with open(file_name, 'r') as f:
        text = f.read()

    return text


def process_text(text, args):
    '''Process text.'''
    sentences = []
    next_chars = []

    # Iterate over text to get sentences and next_chars based on the size of
    # the chunk of text the make up a sentence
    for i in range(0, args.max_length, args.step):
        sentences.append(text[i:i + args.max_length])
        next_chars.append(text[i + args.max_length])

    # Get vocabulary of characters
    chars = sorted(list(set(text)))
    char_idxs = {char: chars.index(char) for char in chars}

    # Prepare nd-arrays of X and Y
    X = np.zeros((len(sentences), args.max_length, len(chars)))
    y = np.zeros((len(sentences), len(chars)))

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_idxs[char]] = 1
            y[i, char_idxs[next_chars[i]]] = 1

    return X, y, chars, char_idxs


def sample(preds, temperature):
    '''
    Reweight probability distribution based on temperature. Higher temperature
    translates to more randomness.
    '''
    preds = np.asarray(preds, dtype='float64')
    preds = np.log(preds) / temperature
    # Normalize to get probs
    probs = np.exp(preds) / np.sum(np.exp(preds))
    # Get random index
    out = np.random.multinomial(1, probs, 1)

    return np.argmax(out)


def build_model(args, num_classes):
    '''Build char-level language model.'''
    # Build and compile the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(args.max_length, num_classes)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-1),
                  loss='categorical_crossentropy')

    return model


def train(model, X_train, y_train, text, char_idxs, chars, args):
    print('Started training ...')

    for epoch in range(args.epochs):
        print(f'{30 * "-":>50s} epoch : {epoch + 1} {30 * "-"}')
        model.fit(X_train,
                  y_train,
                  batch_size=args.batch_size,
                  epochs=1,
                  verbose=args.verbose)

        # Generate random text
        start_idx = np.random.randint(0, len(text) - args.max_length - 1)
        generated_text = text[start_idx:start_idx + args.max_length]

        # Print out out generated text
        sys.stdout.write(generated_text)

        for _ in range(400):
            # Prepare the data to feed it to the model to get predictions
            sampled = np.zeros((1, args.max_length, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_idxs[char]] = 1

            # Get predictions
            preds = model.predict(sampled, verbose=0)[0]
            next_char_idx = sample(preds, args.temperature)
            next_char = chars[next_char_idx]
            generated_text = generated_text[1:]
            generated_text += next_char

            # Print out new chars
            sys.stdout.write(next_char)
        print()
    print('Done training.')

    return model

def main():
    parser = argparse.ArgumentParser(
        description='Training char-level language model using a sample text')
    parser.add_argument('--file_path', type=str, metavar='',
                        help='Path oftext file that will be used in training')
    parser.add_argument('--max_length', type=int, default=50, metavar='',
                        help='Length of sentence used for each sample')
    parser.add_argument('--step', type=int, default=3, metavar='',
                        help='Sliding step used in constructing sentences from'
                             'raw text')
    parser.add_argument('--temperature', type=int, default=1, metavar='',
                        help='Temperature determines the level of randomness'
                             'in generating new text')
    parser.add_argument('--batch_size', type=int, default=128, metavar='',
                        help='Batch size to be used in training')
    parser.add_argument('--epochs', type=int, default=20, metavar='',
                        help='Number of full training cycles')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='Verbosity mode')
    args = parser.parse_args()

    # Read and process text file
    text = read_file(args.file_path)
    X_train, y_train, chars, char_idxs = process_text(text, args)
    
    # Build and train the model
    model = build_model(args, len(chars))
    model = train(model, X_train, y_train, text, char_idxs, chars, args)

    # Save the model on disk
    model.save('models/text-generation.h5')

if __name__ == '__main__':
    main()
