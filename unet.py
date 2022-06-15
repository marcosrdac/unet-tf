# !pip3 install tensorflow_graphics

# +
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models
# import tensorflow_graphics.math.interpolation as tfgmi
# import tensorflow.keras as keras
import keras as K
import numpy as np
from os import environ
# -

# environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.config.run_functions_eagerly(True)


def make_unet(ndim=1,
              input_channels=1,
              output_channels=1,
              scales=5,
              nconvs_by_scale=2,
              base_filters=8,
              kernel_size=3,
              activation='relu',
              first_activation='tanh',
              last_activation='linear',
              norm=False,
              dropout=False,
              norm_at_start=False,
              nconvs_bottom=None,
              use_skip_connections=True,
              return_encoders=False,
              batch_size=None,
              verbose=False):
    '''
    Makes a 1D, 2D or 3D U-net model replacing transposed convolution by resising for decoding.
    '''
    # TODO add dropout?

    if ndim == 1:
        conv = layers.Conv1D
        pool = layers.AveragePooling1D
        drop = layers.SpatialDropout1D
    elif ndim == 2:
        conv = layers.Conv2D
        pool = layers.AveragePooling2D
        drop = layers.SpatialDropout2D
    elif ndim == 3:
        conv = layers.Conv3D
        pool = layers.AveragePooling3D
        drop = layers.SpatialDropout3D
    else:
        raise IndexError('Input data must be 1D, 2D or 3D.')

    nconvs_bottom = nconvs_bottom or nconvs_by_scale
    first_activation = first_activation or activation
    last_activation = last_activation or activation
    
    def convolution(x,
                  filters,
                  kernel_size=kernel_size,
                  strides=1,
                  activation=activation):
        '''Default convolution layer + activation operation.'''
        x = conv(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding='same')(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        return x
    
    def nearest_resample_3d(tensor, shape):
        old_shape = tf.shape(tensor)
        axes = [
            tf.linspace(0, old_shape[0]-1, shape[0]),
            tf.linspace(0, old_shape[1]-1, shape[1]),
            tf.linspace(0, old_shape[2]-1, shape[2]),
            tf.linspace(0, old_shape[3]-1, shape[3]),
        ]
        grid = tf.convert_to_tensor(tf.meshgrid(*axes, indexing='ij'))
        idx = tf.reshape(grid, (4, old_shape[0], -1))
        idx = tf.transpose(idx, (1,2,0))
        idx = tf.cast(idx, tf.int64)
        interp = tf.gather_nd(tensor, idx, batch_dims=0)
        return tf.reshape(interp, shape)

    def encode(x):
        '''Defines U-net encoding phase.'''
        if verbose:
            print('start   ', x.shape)

        if norm_at_start:
            x = layers.BatchNormalization()(x)

        x = convolution(x, base_filters, activation=first_activation)

        if verbose:
            print('prepare ', x.shape)

        old = []

        # downward path
        for scale in range(scales):
            filters = base_filters * 2**(scale + 1)
            for conv in range(nconvs_by_scale):
                x = convolution(x, filters, activation=activation)
            if norm:
                x = layers.BatchNormalization()(x)

            # saving convolution output for skip connections
            old.append(x)

            # lowering resolution
            x = pool(pool_size=3, strides=2, padding='same')(x)
            if verbose:
                print('downward', x.shape)

        # bottom path
        for conv in range(nconvs_bottom):
            x = convolution(x, filters)
        if dropout:
            x = drop(dropout)(x)
        if norm:
            x = layers.BatchNormalization()(x)

        return x, old

    def decode(x, old):
        '''Defines U-net decoding phase.'''
        # upward path
        for scale in range(scales - 1, -1, -1):
            x_old = old[scale]
            filters = int(x_old.shape[-1] / 2)

            # 3D upsample
            x_old_exp = x_old
            for d_miss in range(3 - ndim):
                x = tf.expand_dims(x, axis=3)
                x_old_exp = tf.expand_dims(x_old_exp, axis=3)
            x = nearest_resample_3d(x, tf.shape(x_old_exp))
            for d_miss in range(3 - ndim):
                x = tf.squeeze(x, axis=3+d_miss)

            if use_skip_connections:
                x = layers.Concatenate()([x_old, x])

            for conv in range(nconvs_by_scale):
                x = convolution(x, filters)
            if norm:
                x = layers.BatchNormalization()(x)
            if verbose:
                print('upward  ', x.shape)

        # return correct number of outputs
        x = convolution(x, output_channels, activation=last_activation)

        if verbose:
            print('out     ', x.shape)

        return x

    input_shape = tf.TensorShape(ndim*[None] + [input_channels])
    encoder_input = K.Input(shape=input_shape, batch_size=batch_size, name='input_image')
    encoder_output, old = encode(encoder_input)
    decoder_output = decode(encoder_output, old)

    model = K.Model(encoder_input, decoder_output, name='unet')

    if return_encoders:
        old.append(encoder_output)
        encoders = [
            K.Model(encoder_input, encoder_output, name=f'encoder_{scale}')
            for scale, encoder_output in enumerate(old)
        ]
        return model, encoders
    return model


def make_unet_autoencoder(*args, **kwargs):
    kwargs["output_channels"] = kwargs.get(
        "input_channels", kwargs["input_channels"])
    return make_unet(*args, **kwargs)


if __name__ == "__main__":
    import json
    import pickle as pkl
    from datetime import datetime
    from os.path import dirname, basename, splitext, join
    from os import makedirs
    from tensorflow.keras import optimizers, callbacks
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt


    print('2D U-net')
    X_true = np.ones((10, 32, 64, 3))
    Y_true = np.ones((10, 32, 64, 2))

    print(X_true.ndim-2)
    model_kw = dict(
        ndim=X_true.ndim - 2, # -(batch_dims+channel_dims)
        input_channels=X_true.shape[-1],
        output_channels=Y_true.shape[-1],
        scales=3,
        nconvs_by_scale=2,
        base_filters=8,
        kernel_size=3,
        activation='relu',
        first_activation='tanh',
        last_activation='linear',
        dropout=.3,
        norm=False,
        norm_at_start=False,
        nconvs_bottom=None,
        use_skip_connections=True,
    )

    NOW = f'{datetime.now():%Y%m%dT%H%M}'
    model = make_unet(**model_kw, verbose=True)
    n_params = model.count_params()
    n_trainable_params = np.sum([np.size(w) for w in model.trainable_weights])

    Y_pred = model.predict(X_true)
    print(f'{X_true.shape = }')
    print(f'{Y_true.shape = }')
    print(f'{Y_pred.shape = }')
    print(f'n_params = {n_params}\n')
    print(f'n_trainable_params = {n_trainable_params}\n')

    # model.summary()

    # holdout validation
    X_train, X_test, Y_train, Y_test = train_test_split(X_true,
                                                        Y_true,
                                                        test_size=1 / 4)
    print('n_train: {X_train.shape[0]}')
    print('n_test:  {X_test.shape[0]}\n')

    # training model
    learning_rate = 0.0005
    max_epochs = 4
    batch_size = X_train.shape[0]
    # batch_size = 1

    CHECKPOINT_DIR = 'model_checkpoints'
    makedirs(CHECKPOINT_DIR, exist_ok=True)

    optim = optimizers.Adam(learning_rate)
    model.compile(optimizer=optim, loss='mae', metrics=[])

    callback_list = [
        callbacks.ModelCheckpoint(join(CHECKPOINT_DIR, 'weights.h5'),
                                  save_best_only=True),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            min_delta=0,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=True,
        )
    ]

    train = model.fit(X_train,
                      Y_train,
                      epochs=max_epochs,
                      validation_data=(X_test, Y_test),
                      callbacks=callback_list,
                      batch_size=batch_size)

    # calculating loss and defining name under which to save model
    val_loss_min = np.min(train.history['val_loss'])
    name = f"{NOW}_unet_loss={val_loss_min:.4g}_nparams={n_params}"

    # plotting data
    FIGURE_DIR = 'figures'
    makedirs(FIGURE_DIR, exist_ok=True)

    fig_path = join(FIGURE_DIR, f'{name}_loss_history.png')
    fig, ax = plt.subplots()
    epochs = 1 + np.arange(len(train.history['loss']))
    ax.plot(epochs, train.history['loss'], label='Train')
    ax.plot(epochs, train.history['val_loss'], label='Test')
    # ax.yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid()
    ax.axvline(
        epochs[np.argmin(train.history['val_loss'])],
        label='Best model',
        c='k',
        ls='--',
    )
    ax.legend()
    fig.savefig(fig_path)
    plt.show()

    # saving model stuff
    MODEL_DIR = 'models'
    makedirs(MODEL_DIR, exist_ok=True)

    # json model
    with open(join(MODEL_DIR, f'{name}_model.json'), 'w') as f:
        f.write(model.to_json())
    # model weights
    model.save_weights(join(MODEL_DIR, f'{name}_weights.h5'))
    # json model keywords
    with open(join(MODEL_DIR, f'{name}_description.json'), 'w') as f:
        json.dump(model_kw, f)
    # history
    with open(join(MODEL_DIR, f'{name}_history.pkl'), 'wb') as f:
        pkl.dump(train.history, f)
