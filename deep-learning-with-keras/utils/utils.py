import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras.backend as K


def plot_loss_and_metric(history, metric_name='accuracy'):
    '''Plot training and validation loss and metric on two grids.'''
    acc = history.history[metric_name]
    loss = history.history['loss']
    val_acc = history.history['val_' + metric_name]
    val_loss = history.history['val_loss']
    epochs = len(acc)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(range(1, epochs + 1), loss, label='Training loss')
    axes[0].plot(range(1, epochs + 1), val_loss, label='Validation loss')
    axes[0].set_xlabel('Iteration', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].set_title('Training and validation loss', fontsize=20)
    axes[0].legend(fontsize=14)
    axes[1].plot(range(1, epochs + 1), acc, label='Training accuracy')
    axes[1].plot(range(1, epochs + 1), val_acc, label='Validation accuracy')
    axes[1].set_xlabel('Iteration', fontsize=18)
    axes[1].set_ylabel('Accuracy', fontsize=18)
    axes[1].set_title('Training and validation accuracy', fontsize=20)
    axes[1].legend(fontsize=14)
    plt.tight_layout()


def feature_extraction(directory, conv_base, num_examples, batch_size=20):
    '''Compute extracted features using `conv_base` of pretrained model.'''
    data_gen = ImageDataGenerator(rescale=1/255)
    generator = data_gen.flow_from_directory(directory,
                                             target_size=(150, 150),
                                             batch_size=batch_size,
                                             class_mode='binary')
    features_extracted = np.zeros((num_examples, 4, 4, 512))
    labels = np.zeros((num_examples))

    i = 0
    for batch_input, batch_label in generator:
        features = conv_base.predict(batch_input)
        features_extracted[i * batch_size:(i + 1) * batch_size] = features
        labels[i * batch_size:(i + 1) * batch_size] = batch_label
        i += 1
        if i * batch_size >= num_examples:
            break

    return features_extracted, labels


def plot_conv_outputs(model, img_path, images_per_row=16):
    '''
    Plot output of convolutional layers (activations) in a CNN.
    '''
    # Convert the image into tensor
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.reshape(img, ((1,) + img.shape))
    img /= 255

    # Get the the activations
    activations = model.predict(img)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[
                    0, :, :, col * images_per_row + row
                    ]

                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


def generate_patterns(model, layer_name, filter_index=0, iterations=50):
    '''
    Return what an image-like tensor that shows what the `filter_index` in 
    `layer_name` is most responsive to.
    '''
    # Generate random gray image
    random_img = np.random.random((1, 150, 150, 3)) * 20 + 128

    # Define output and loss
    output = model.get_layer(layer_name).output
    loss = K.mean(output[:, :, :, filter_index])

    # Define gradients
    grads = K.gradients(loss, model.input)[0]
    # Normalize gradients
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Define a function that returns loss and gradients w.r.t. input
    compute_loss_grads = K.function([model.input], [loss, grads])

    # Start the gradient ascent steps
    lr = 1
    for i in range(iterations):
        loss_value, grads_value = compute_loss_grads([random_img])
        random_img += lr * grads_value

    img = random_img[0]
    
    return deprocess_image(img)


def smooth_curve(points, factor=0.9):
    '''Add smoothness to set of points.'''
    smooth_points = []

    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(factor * previous + (1 - factor) * point)

        else:
            smooth_points.append(point)

    return smooth_points


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x
