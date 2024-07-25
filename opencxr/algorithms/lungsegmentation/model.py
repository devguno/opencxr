# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    Dropout,
    UpSampling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model

def unet(
    x_in,
    k_size=3,
    optimizer="adam",
    depth=4,
    downsize_filters_factor=1,
    batch_norm=False,
    activation="relu",
    initializer="glorot_uniform",
    seed=42,
    upsampling=False,
    dropout=[(-1, 0.5)],
    n_convs_per_layer=2,
    lr=False,
    loss="binary",
):
    # Fix the seed into given argument.
    np.random.seed(seed)

    # if True, deepest layer with 0.5 probability.
    if dropout == True:
        dropout = [(-1, 0.5)]

    # Define the hyperparameter settings.
    # Max number of init filters: 64
    settings = {
        "n_classes": 2,  # nb of classes always 2.
        "depth": depth,  # unet depth 4
        "filters": 64 / downsize_filters_factor,  # unet filters 64
        "kernel_size": (k_size, k_size),  # unet kernel size (3,3)
        "pool_size": (2, 2),  # standard 2,2
        "n_convs_per_layer": n_convs_per_layer,  # standard perlayer 2 conv.
        "activation": activation,  # relu standard
        "kernel_initializer": initializer,  # he_normal standard
        "padding": "same",  # valid standard
        "dropout": dropout,  # dropout is the deepest layer.
        "batch_norm": batch_norm,  # if to use batch norm or not.
        "upsampling": upsampling,  # if true upsampling, else conv2dtranspose.
    }

    data = Input(shape=x_in)  # input layer
    layers = {}  # hold all the layers in a dictionary.
    l = data  # input dimension.

    def conv(filters):
        return Conv2D(
            filters=filters,
            kernel_size=settings["kernel_size"],
            activation=settings["activation"],
            kernel_initializer=settings["kernel_initializer"],
            padding=settings["padding"],
        )

    def dropout(rate):
        return Dropout(rate)

    def batchnorm():
        return BatchNormalization()

    def pool():
        return MaxPooling2D(pool_size=settings["pool_size"])

    def concat():
        return Concatenate()

    def t_conv(filters):
        if upsampling:
            return UpSampling2D(size=settings["pool_size"])
        else:
            return Conv2DTranspose(
                filters=filters,
                kernel_size=settings["pool_size"],
                strides=settings["pool_size"],
                kernel_initializer=settings["kernel_initializer"],
                padding=settings["padding"],
            )

    def add(layer, l_in, name):
        layers[name] = layer(l_in)
        return layers[name]

    depths = list(range(settings["depth"]))

    if settings["dropout"] != False:
        dropout_depths = list(range(settings["depth"] + 1))
        dropouts = {dropout_depths[d]: rate for d, rate in settings["dropout"]}

    contracting_outputs = {}
    for i in depths:
        for j in range(settings["n_convs_per_layer"]):
            n = int(settings["filters"] * (2 ** i))
            l = add(conv(n), l, "conv_down_{}_{}".format(i, j))
            contracting_outputs[i] = l

            if settings["batch_norm"] == True:
                l = add(batchnorm(), l, "batch_norm_{}_{}".format(i, j))
                contracting_outputs[i] = l

            if settings["dropout"] != False:
                if i in dropouts:
                    l = add(dropout(dropouts[i]), l, "dropout_{}_{}".format(i, j))
                    contracting_outputs[i] = l

        l = add(pool(), l, "pool_{}".format(i))

    i = settings["depth"]
    for j in range(settings["n_convs_per_layer"]):
        n = int(settings["filters"] * 2 ** settings["depth"])
        l = add(conv(n), l, "conv_{}_{}".format(settings["depth"], j))

        if settings["dropout"] != False:
            if i in dropouts:
                l = add(dropout(dropouts[i]), l, "dropout_{}_{}".format(i, j))

    for i in reversed(depths):
        n = int(settings["filters"] * 2 ** i)
        l = add(t_conv(n), l, "t_conv{}".format(i))
        l = add(concat(), [l, contracting_outputs[i]], "concat_{}".format(i))
        for j in range(settings["n_convs_per_layer"]):
            n = int(settings["filters"] * 2 ** i)
            l = add(conv(n), l, "conv_up_{}_{}".format(i, j))

            if settings["batch_norm"] == True:
                l = add(batchnorm(), l, "batch_norm_{}_{}".format(i, j))
                contracting_outputs[i] = l

    lr = float(lr)
    if lr != "default":
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, decay=1e-3)
        elif optimizer == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True
            )
    else:
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True
            )

    if loss == "binary":
        out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")
        l = add(out, l, "out")
        model = Model(data, l)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
    else:
        out = Conv2D(filters=2, kernel_size=(1, 1), activation="softmax")
        l = add(out, l, "out")
        model = Model(data, l)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    return model
