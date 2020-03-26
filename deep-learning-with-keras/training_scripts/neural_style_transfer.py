'''
Using VGG19 pretrained model on imagenet to.
'''


from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    
    return img


def deprocess_image(img):
    # Reverse preprocess done by vgg19
    img[:, :, 0] += 103.99
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Convert image from BGR to RGB
    img = img[:, :, ::-1]
    
    return img


def content_loss(base, generated):
    loss = K.sum(K.square(generated - base))
    
    return loss


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, [2, 0, 1]))
    gram = K.dot(features, K.transpose(features))
    
    return gram


def style_loss(style_img, generated_img):
    S = gram_matrix(style_img)
    G = gram_matrix(generated_img)
    loss = (1 / ((2 * img_height * img_width * 3) ** 2)) * K.sum(K.square(S - G))
    
    return loss


for layer in style_layers:
    layer_features = output_dict[layer]
    style_img_features = layer_features[1, :, :, :]
    generated_img_features = layer_features[2, :, :, :]
    loss += (style_weight / len(style_layers)) * style_loss(style_img_features, generated_img_features)


class Evaluator:
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_gradients([x])
        loss_value = outs[0]
        grads_values = outs[1].flatten().astype("float32")
        self.loss_value = loss_value
        self.grads_values = grads_values
        
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        
        return grad_values