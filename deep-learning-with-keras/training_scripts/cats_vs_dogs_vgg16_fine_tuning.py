'''
Fine tune VGG16 ConvNet trained on imagenet to make it more relevant to our
dataset cats-vs-dogs. Steps for training:
1. Add conv_base as a first layer to the model.
2. Add classifier on top of conv_base layer.
3. Freeze conv_base layer so that all the weights of it become not trainable.
4. Train the model with relatively low lr.
5. Unfreeze last block of conv_base layer and recompile the model.
6. Retrain the model with also low lr to not have huge changes to the weights
of the unfreezed block of conv_base.
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


def train(args):
    # Build data generators
    train_data_gen = ImageDataGenerator(rescale=1/255,
                                        horizontal_flip=True,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        rotation_range=40,
                                        zoom_range=0.2,
                                        shear_range=0.2,
                                        fill_mode='nearest')
    test_data_gen = ImageDataGenerator(rescale=1/255)
    train_gen = train_data_gen.flow_from_directory(args.train_dir,
                                                   target_size=(150, 150),
                                                   batch_size=args.batch_size,
                                                   class_mode='binary')
    valid_gen = test_data_gen.flow_from_directory(args.valid_dir,
                                                  target_size=(150, 150),
                                                  batch_size=args.batch_size,
                                                  class_mode='binary')

    # Build conv_base
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(150, 150, 3))

    # Build model
    model = models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Freeze conv_base trainabel weights
    print(f'Number of trainable weights before freezing conv_base weights: '
          f'{len(model.trainable_weights)}')
    conv_base.trainable = False
    print(f'Number of trainable weights after freezing conv_base weights: '
          f'{len(model.trainable_weights)}')

    # Compile model
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_dir_size = len(os.listdir(os.path.join(
                            args.train_dir, os.listdir(args.train_dir)[0])))
    valid_dir_size = len(os.listdir(os.path.join(
                            args.valid_dir, os.listdir(args.valid_dir)[0])))
    steps_per_epoch = (2 * train_dir_size) / args.batch_size
    validation_steps = (2 * valid_dir_size) / args.batch_size

    # Fit the model
    print('Start training with freezed conv_base....')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.epochs,
                                  validation_data=valid_gen,
                                  validation_steps=validation_steps,
                                  verbose=args.verbose)
    print('Training is done.')

    # Unfreeze last block in conv_base
    for layer in conv_base.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True

    # Recompile model
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Fit the model
    print('Fine tuning with last block of conv_base unfreezed....')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.epochs,
                                  validation_data=valid_gen,
                                  validation_steps=validation_steps,
                                  verbose=args.verbose)
    print('Fine tuning is done.')

    # Save model
    if not os.path.exists('models/'):
        os.mkdir('models')
    model.save('models/cats-vs-dogs-samll-vgg16-fine-tune.h5')

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

    # train classifier and test it
    model = train(args)
    test(args, model)


if __name__ == '__main__':
    main()
