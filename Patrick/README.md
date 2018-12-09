# Patrick's implementations

## Super Resolution CNN (SRCNN)
_________________________________________________________________
Layer (type)                 Output Shape              Param
=================================================================
input_1 (InputLayer)         (None, 128, 128, 1)       0
_________________________________________________________________
conv1 (Conv2D)               (None, 128, 128, 64)      5248
_________________________________________________________________
conv2 (Conv2D)               (None, 128, 128, 32)      51232
_________________________________________________________________
conv3 (Conv2D)               (None, 128, 128, 1)       801
=================================================================
Total params: 57,281
Trainable params: 57,281
Non-trainable params: 0
_________________________________________________________________

## Subpixel + Super Resolution CNN (Subpixel + SRCNN)

## Expanded Super Resolution CNN (ESRCNN)

## Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)

## My Test Network