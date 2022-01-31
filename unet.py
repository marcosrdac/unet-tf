import json
import keras as K
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from datetime import datetime
from os.path import dirname, basename, splitext, join
from os import makedirs
from os import environ

# environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.config.run_functions_eagerly(True)


def make_unet(input_shape,
              nout=1,
              scales=5,
              nconvs_by_scale=2,
              base_filters=8,
              kernel_size=3,
              activation='relu',
              first_activation='tanh',
              last_activation='linear',
              interpolator='nearest',
              last_interpolator=None,
              norm=False,
              dropout=False,
              norm_at_start=False,
              nconvs_bottom=None,
              use_skip_connections=True,
              return_encoders=False,
              verbose=False):
    '''
    Makes 1D or 2D U-net model with resising instead of upsampling layers when
    decoding.
    '''
    # TODO add dropout?

    *resolution, feats = input_shape
    ndim = len(resolution)

    if ndim == 1:
        Conv = layers.Conv1D
        Pooling = layers.AveragePooling1D
        SpatialDropout = layers.SpatialDropout1D
    elif ndim == 2:
        Conv = layers.Conv2D
        Pooling = layers.AveragePooling2D
        SpatialDropout = layers.SpatialDropout2D
    else:
        raise IndexError('Input data must be 1D or 2D.')

    nconvs_bottom = nconvs_bottom or nconvs_by_scale
    first_activation = first_activation or activation
    last_activation = last_activation or activation
    last_interpolator = last_interpolator or interpolator

    def ConvLayer(x,
                  filters,
                  kernel_size=kernel_size,
                  strides=1,
                  activation=activation):
        '''Default convolution layer + activation operation.'''
        x = Conv(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding='same')(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        return x

    def encode(x):
        '''Defines U-net encoding phase.'''
        if verbose:
            print('start   ', x.shape)

        if norm_at_start:
            x = layers.BatchNormalization()(x)

        x = ConvLayer(x, base_filters, activation=first_activation)

        if verbose:
            print('prepare ', x.shape)

        old = []

        # downward path
        for scale in range(scales):
            filters = base_filters * 2**(scale + 1)
            for conv in range(nconvs_by_scale):
                x = ConvLayer(x, filters, activation=activation)
            if norm:
                x = layers.BatchNormalization()(x)

            # saving convolution output for skip connections
            old.append(x)

            # lowering resolution
            x = Pooling(pool_size=3, strides=2, padding='same')(x)
            if verbose:
                print('downward', x.shape)

        # bottom path
        for conv in range(nconvs_bottom):
            x = ConvLayer(x, filters)
        if dropout:
            x = SpatialDropout(dropout)(x)
        if norm:
            x = layers.BatchNormalization()(x)

        return x, old

    def decode(x, old):
        '''Defines U-net decoding phase.'''
        # upward path
        for scale in range(scales - 1, -1, -1):
            x_old = old[scale]
            filters = int(x_old.shape[-1] / 2)

            # 'if' needed for 1D/2D data resizing compatibility
            if ndim == 1:
                x = x[:, :, None, :]
                x_old = x_old[:, :, None, :]
            _interpolator = last_interpolator if scale == 0 else interpolator
            x = layers.Resizing(*x_old.shape[1:-1], _interpolator)(x)
            if ndim == 1:
                x = x[:, :, 0, :]
                x_old = x_old[:, :, 0, :]

            if use_skip_connections:
                x = layers.Concatenate()([x_old, x])

            for conv in range(nconvs_by_scale):
                x = ConvLayer(x, filters)
            if norm:
                x = layers.BatchNormalization()(x)
            if verbose:
                print('upward  ', x.shape)

        # final convolution return correct number of outputs
        x = ConvLayer(x, nout, activation=last_activation)

        if verbose:
            print('out     ', x.shape)

        return x

    encoder_input = K.Input(shape=input_shape, name='input_image')
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


def make_unet_autoencoder(input_shape, *args, **kwargs):
    return make_unet(input_shape, nout=input_shape[-1], *args, **kwargs)


if __name__ == "__main__":
    # defining data
    # print('1D U-net')
    # X_true = np.ones((10, 64, 3))
    # Y_true = np.ones((10, 64, 2))

    print('2D U-net')
    X_true = np.ones((10, 32, 64, 3))
    Y_true = np.ones((10, 32, 64, 2))

    model_kw = dict(
        input_shape=X_true.shape[1:],
        nout=Y_true.shape[-1],
        scales=5,
        nconvs_by_scale=2,
        base_filters=8,
        kernel_size=3,
        activation='relu',
        first_activation='tanh',
        last_activation='linear',
        interpolator='nearest',
        last_interpolator=None,
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
