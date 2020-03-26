import os
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


def show_images(images, mean, std, figsize=(12, 12), title=None):
    '''Plot image from a torch tensor.'''
    img = images.to('cpu').clone().detach()
    img = img.numpy()
    img = img.transpose([1, 2, 0])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    if title is not None:
        plt.title(title)


def plot_loss_and_metric(history, figsize=(16, 8)):
    '''Plot training and validation loss and metric on two grids.'''
    epochs = range(1, len(history['loss']['train']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, k in zip(axes, history.keys()):
        ax.plot(epochs, history[k]['train'], label=f'Training {k}')
        ax.plot(epochs, history[k]['valid'], label=f'Validation {k}')
        ax.set_xlabel('Epoch', {'fontsize': 14})
        ax.set_ylabel(k.capitalize(), {'fontsize': 14})
        ax.set_title(f'Training and validation {k}', {'fontsize': 16})
        ax.legend()
    plt.tight_layout()


def load_img(img, mean, std):
    '''Preprocess image tensor'''
    img = img.numpy()
    img = img.transpose([1, 2, 0])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def rand_by_mask(mask, num_images):
    '''Get random indices using mask'''
    idxs = np.where(mask)[0]
    rand_idxs = np.random.choice(idxs, num_images)
    return rand_idxs


def rand_by_correct(is_correct, correct_labels_val, num_images=4):
    '''Get random indices for `is_correct` predictions'''
    idxs = rand_by_mask(correct_labels_val == is_correct, num_images)
    return idxs


def most_by_mask(mask, is_correct, class_, probs_val, num_images):
    '''Get most indices using mask'''
    idxs = np.where(mask)[0]
    if is_correct:
        sorted_idxs = np.argsort(probs_val[:, class_][idxs])[::-1] # Descending
    else:
        sorted_idxs = np.argsort(probs_val[:, class_][idxs]) # Ascending
    return idxs[sorted_idxs[:num_images]]


def most_by_correct(class_, is_correct, data_val_y, correct_labels_val,
                    probs_val, num_images=4):
    '''
    Get most accurate/inaccurate indices for `class_` and `is_accurate`
    predictions
    '''
    mask = ((correct_labels_val == is_correct) & (data_val_y == class_))
    return most_by_mask(mask, is_correct, class_, probs_val, num_images)


def plot_val_images(idxs, data_val_imgs, data_val_y, probs_val, mean,
                    std, num_images=4, figsize=(12, 12), title='test'):
    f = plt.figure(figsize=figsize)
    for i, idx in enumerate(idxs):
        f.add_subplot(1, num_images, i + 1)
        img = load_img(data_val_imgs[idx], mean, std)
        label = data_val_y[idx]
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{probs_val[idx][label]:.3f}')
    print(title)


def find_lr(model, device, criterion, optimizer, init_lr, final_lr, train_dataloader,
            beta=0.98):
    '''
    Implementing learning rate finder from Lelie Smith paper.
    '''
    n = len(train_dataloader)
    avg_loss = 0
    best_loss = 0
    smoothed_losses = []
    log_lr = []
    factor = (final_lr / init_lr) ** (1 / (n - 1))
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr

    for batch_num, data in enumerate(train_dataloader, 1):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Fwd pass
        output = model(inputs)
        loss = criterion(output, labels)
        # Compute average & smoothed losses
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if loss > 4 x best loss
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lr, smoothed_losses
        # Update the best loss
        if batch_num == 1 or smoothed_loss < best_loss:
            best_loss = smoothed_loss
        # Record the values
        smoothed_losses.append(smoothed_loss)
        log_lr.append(np.log10(lr))
        # Bckwd pass
        loss.backward()
        optimizer.step()
        # Update learning rate
        lr *= factor
        optimizer.param_groups[0]['lr'] = lr

    return log_lr, smoothed_losses


def validate(net, valid_dataloader):
    '''Run the model on validation data.'''
    with torch.no_grad():
        i = 0
        for images, labels in valid_dataloader:
            out = net(images)
            probs = nn.Softmax(dim=1)(out)
            preds = torch.argmax(probs, 1)
            correct_labels = labels == preds
            if i == 0:
                probs_val = probs.numpy()
                correct_labels_val = correct_labels.numpy()
                data_val_imgs = images
                data_val_y = labels
            else:
                probs_val = np.concatenate([probs_val, probs.numpy()])
                correct_labels_val = np.concatenate([correct_labels_val, correct_labels.numpy()])
                data_val_imgs = torch.cat([data_val_imgs, images])
                data_val_y = torch.cat([data_val_y, labels])
            i += 1
    valid = {
        'probs_val': probs_val,
        'correct_labels_val': correct_labels_val,
        'data_val_imgs': data_val_imgs,
        'data_val_y': data_val_y
        }
    return valid
