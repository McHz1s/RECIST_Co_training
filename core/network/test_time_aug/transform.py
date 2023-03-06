import torch as tf


def identity(img):
    return img


def hflip(img):
    return tf.flip(img, dims=[-2])


def vflip(img):
    return tf.flip(img, dims=[-1])
