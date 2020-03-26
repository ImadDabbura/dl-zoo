'''
Train a classifier NN using the CNN base of VGG16 as a feature extraction
module on cats-vs-dogs dataset. Steps of training:
1. Run train, valid, and test images through conv_base and save them on desk.
2. Train the classifier on the output from conv_base.
3. Evaluate the model using test features and labels obtained from conv_base.
Dataset can be found here: https://www.kaggle.com/c/dogs-vs-cats.
'''


import os
from argparse import ArgumentParser

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras import models, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')


def extract_features(directory, sample_size, conv_base, batch_size=20):
    # Instatiate features and labels
    features = np.zeros((sample_size, 4, 4, 512))
    labels = np.zeros(sample_size)

    # Define generators
    data_gen = ImageDataGenerator(rescale=1/255)
    dir_gen = data_gen.flow_from_directory(directory,
                                           target_size=(150, 150),
                                           batch_size=batch_size,
                                           class_mode='binary')

    # Used to determine when to break the loop
    i = 0
    # Pass images through conv_base to get extracted features
    for images, images_labels in dir_gen:
        features[i * batch_size:(i + 1) * batch_size] = conv_base.predict(images)
        labels[i * batch_size:(i + 1) * batch_size] = images_labels

        i += 1
        if i * batch_size >= sample_size:
            break

    return features, labels


def train(args, train_features, train_labels, valid_features, valid_labels):
    # Build model
    model = models.Sequential()
    model.add(Dense(256, activation='relu', input_shape=(512 * 4 * 4,)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Fit the model
    print('Start training ....')
    history = model.fit(train_features,
                        train_labels,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=[valid_features, valid_labels],
                        verbose=args.verbose)
    print('Training is done.')

    # Save model
    if not os.path.exists('models/'):
        os.mkdir('models')
    model.save('models/cats-vs-dogs-samll-vgg16-feat-ext-no-aug.h5')

    return model


def test(args, model, test_features, test_labels):
    loss, accuracy = model.evaluate(test_features, test_labels)
    print(f'The test accuracy is : {accuracy:.2%}.')


def main():
    parser = ArgumentParser(description='Train a CNN on Cats-vs-Dogs dataset')
    parser.add_argument('--train_dir', type=str, metavar='',
                        help='Directory containing training images')
    parser.add_argument('--valid_dir', type=str, metavar='',
                        help='Directory containing validation images')
    parser.add_argument('--test_dir', type=str, metavar='',
                        help='Directory containing test images')
    parser.add_argument('--batch_size', type=int, default=20, metavar='',
                        help='Batch size to be used when training the CNN')
    parser.add_argument('--epochs', type=int, default=20, metavar='',
                        help='Number of full training cycles')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='Verbosity mode')
    args = parser.parse_args()

    # Build conv_base
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(150, 150, 3))

    # Get extracted features
    train_features, train_labels = extract_features(
        args.train_dir, 2000, conv_base, 20)
    valid_features, valid_labels = extract_features(
        args.valid_dir, 1000, conv_base, 20)
    test_features, test_labels = extract_features(
        args.test_dir, 1000, conv_base, 20)

    # Reshape features
    train_features = np.reshape(train_features, (2000, -1))
    valid_features = np.reshape(valid_features, (1000, -1))
    test_features = np.reshape(test_features, (1000, -1))

    # train classifier and test it
    model = train(
        args, train_features, train_labels, valid_features, valid_labels)
    test(args, model, test_features, test_labels)


if __name__ == '__main__':
    main()
