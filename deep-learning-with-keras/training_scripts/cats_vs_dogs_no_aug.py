'''
Train simple CNN on cats-vs-dogs dataset with no data augmentation.
Dataset can be found here: https://www.kaggle.com/c/dogs-vs-cats.
'''


import os
from argparse import ArgumentParser

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras import models
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from warnings import filterwarnings
filterwarnings('ignore')


def train(args):
    # Build data generators
    data_gen = ImageDataGenerator(rescale=1/255)
    train_gen = data_gen.flow_from_directory(args.train_dir,
                                             target_size=(150, 150),
                                             batch_size=args.batch_size,
                                             class_mode='binary')
    valid_gen = data_gen.flow_from_directory(args.valid_dir,
                                             target_size=(150, 150),
                                             batch_size=args.batch_size,
                                             class_mode='binary')

    # Build model
    model = models.Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPool2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_dir_size = len(os.listdir(os.path.join(
                            args.train_dir, os.listdir(args.train_dir)[0])))
    valid_dir_size = len(os.listdir(os.path.join(
                            args.valid_dir, os.listdir(args.valid_dir)[0])))
    steps_per_epoch = (2 * train_dir_size) / args.batch_size
    validation_steps = (2 * valid_dir_size) / args.batch_size

    # Fit the model
    print('Start training ....')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.epochs,
                                  validation_data=valid_gen,
                                  validation_steps=validation_steps,
                                  verbose=args.verbose)
    print('Training is done.')

    # Save model
    if not os.path.exists('models/'):
        os.mkdir('models')
    model.save('models/cats-vs-dogs-samll-no-aug.h5')

    return model


def test(args, model):
    # Build data generators
    data_gen = ImageDataGenerator(rescale=1/255)
    test_gen = data_gen.flow_from_directory(args.test_dir,
                                            target_size=(150, 150),
                                            batch_size=args.batch_size,
                                            class_mode='binary')
    loss, accuracy = model.evaluate_generator(test_gen)
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

    model = train(args)
    test(args, model)


if __name__ == '__main__':
    main()
