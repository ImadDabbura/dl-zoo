'''
Copy cats and dogs images from original downloaded dataset directory to
three directories: train, validation and test directory.
'''


import os
import shutil


def copy_cats_and_dogs_images(
        src_dir, dst_dir, train_size, validation_size, test_size):
    '''
    Copy cats and dogs images from original downloaded dataset directory to
    three directories: train, validation and test directory.

    parameters
    ----------
    src_dir : str (required)
        source directory starting with the root.
    dst_dir : str (rquired)
        destination directory starting with the root.
    train_size : int (required)
        size of training set.
    validation_size : int (required)
        size of validation set.
    test_size : int (required)
        size of test set.

    Return
    ------
    image_dir : tuple
        three main image directories: train, validation and test.
    '''
    assert (train_size > 0) & (validation_size > 0) & (test_size > 0), \
        ('All train, validation and test sizes should be strictly > 0')

    # Set the limit of validation and test sets
    valid_lim = train_size + validation_size
    test_lim = valid_lim + test_size

    # Set train, validation and test directories
    train_dir = os.path.join(dst_dir, 'train')
    test_dir = os.path.join(dst_dir, 'test')
    validation_dir = os.path.join(dst_dir, 'validation')

    # All directories for cats images
    train_cats_dir = os.path.join(train_dir, 'cats')
    test_cats_dir = os.path.join(test_dir, 'cats')
    validation_cats_dir = os.path.join(validation_dir, 'cats')

    # All directories for dogs images
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    # Copy cats images to their respective directories
    # Copy the first 1000 images of cats to training directory
    fnames = [f'cat.{i}.jpg' for i in range(train_size)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy the next 500 images of cats to validation directory
    fnames = [f'cat.{i}.jpg' for i in range(train_size, valid_lim)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy the next 500 images of cats to test directory
    fnames = [f'cat.{i}.jpg' for i in range(valid_lim, test_lim)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copy dogs images to their respective directories
    # Copy the first 1000 images of dogs to training directory
    fnames = [f'dog.{i}.jpg' for i in range(train_size)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Copy the next 500 images of dogs to validation directory
    fnames = [f'dog.{i}.jpg' for i in range(train_size, valid_lim)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Copy the next 500 images of dogs to test directory
    fnames = [f'dog.{i}.jpg' for i in range(valid_lim, test_lim)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Printing the number of images for each subdirectory
    print(f'Training cat images: {len(os.listdir(train_cats_dir))}.')
    print(f'Validation cat images: {len(os.listdir(validation_cats_dir))}.')
    print(f'Test cat images: {len(os.listdir(test_cats_dir))}.')
    print(f'Training dog images: {len(os.listdir(train_dogs_dir))}.')
    print(f'Validation dog images: {len(os.listdir(validation_dogs_dir))}.')
    print(f'Test dog images: {len(os.listdir(test_dogs_dir))}.')

    image_dir = (train_dir, validation_dir, test_dir)

    return image_dir
