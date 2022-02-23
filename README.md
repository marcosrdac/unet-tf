# U-Net implementation

1D/2D keras U-net model.

## Hyper-parameter definition example

```
unet = make_unet(input_shape=(64,64, 3),
                 nout=1,  # number of channels in output
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
                 verbose=False)

Y = unet.predict(X)
```
