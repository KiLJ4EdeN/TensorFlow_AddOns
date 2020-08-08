# BASIC BLOCKS
import tensorflow as tf
from tensorflow.keras.constraints import max_norm

def __conv_block(x, filter_start=64, kernel_size=(2, 2),
                 num_blocks=2,
                 use_bn=True, use_constraint=True,
                 use_dropout=True, constraint_rate=1,
                 dropout_rate=0.25, activation='relu'):
  # simple convolutional block.
  if use_constraint:
    constraint = max_norm(constraint_rate)
  else:
    constraint = None

  for i in range(num_blocks):
    x = tf.keras.layers.Conv2D(filters=filter_start*(i+1), kernel_size=kernel_size,
                               kernel_constraint=constraint)(x)
    if use_bn:
      x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=filter_start*(i+1), kernel_size=kernel_size,
                               kernel_constraint=constraint)(x)
    if use_bn:
      x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(filters=filter_start*(i+1), kernel_size=(1, 1),
                               strides=(2, 2))(x)
    if use_dropout:
      x = tf.keras.layers.Dropout(dropout_rate)(x)
  return x

def __dense_block(x, unit_start=128, num_blocks=2,
                  flatten=True, use_constraint=True,
                  use_dropout=True, constraint_rate=1,
                  dropout_rate=0.5, activation='relu'):
  # simple mlp structure.
  if use_constraint:
    constraint = max_norm(constraint_rate)
  else:
    constraint = None

  if flatten:
    x = tf.keras.layers.Flatten()(x)

  for i in range(num_blocks):
    x = tf.keras.layers.Dense(units=unit_start*(i+1),
                              kernel_constraint=constraint)(x)
    x = tf.keras.layers.Activation(activation)(x)
    if use_dropout:
      x = tf.keras.layers.Dropout(dropout_rate)(x)
  return x

def __classification_block(x, num_classes=100):
  # classification head.
  x = tf.keras.layers.Dense(units=num_classes)(x)
  x = tf.keras.layers.Activation('softmax')(x)
  return x
  
  
# ADVANCED BLOCKS
import tensorflow as tf
from tensorflow.keras.constraints import max_norm

def __parallel_block(x, width=2, filter_start=64,
                     num_blocks=2,
                     use_bn=True, use_constraint=True,
                     use_dropout=True, constraint_rate=1,
                     dropout_rate=0.3, activation='relu'):
  # parallel architecture.
  branches = []
  for i in range(width):
    f = __conv_block(x, filter_start=filter_start,
                     kernel_size=(i+2, i+2),
                     num_blocks=num_blocks,
                     use_bn=use_bn, use_constraint=use_constraint,
                     use_dropout=use_dropout, constraint_rate=constraint_rate,
                     dropout_rate=dropout_rate, activation=activation)
    f = tf.keras.layers.Flatten()(f)
    branches.append(f)
  x = tf.keras.layers.concatenate(branches)
  return x

def __residual_block(x, filter_start=64, kernel_size=(2, 2),
                     use_bn=True, use_constraint=False,
                     use_dropout=False, constraint_rate=1,
                     dropout_rate=0.25, activation='relu'):
  # residual block.
  if use_constraint:
    constraint = max_norm(constraint_rate)
  else:
    constraint = None
    
  x_shortcut = x
  # Path 1
  x = tf.keras.layers.Conv2D(filter_start*1,
                             kernel_size=(1, 1),
                             strides=(2, 2),
                             padding='valid')(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  # Path 2
  x = tf.keras.layers.Conv2D(filters=filter_start*2,
                             kernel_size=kernel_size, 
                             strides=(1, 1), padding='same',
                             kernel_constraint=constraint)(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  # Path 3
  x = tf.keras.layers.Conv2D(filters=filter_start*4,
                             kernel_size=(1, 1),
                             strides=(1, 1), padding='valid',
                             kernel_constraint=constraint)(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  # Shortcut
  x_shortcut = tf.keras.layers.Conv2D(filters=filter_start*4,
                                      kernel_size=(1, 1),
                                      strides=(2, 2),
                                      padding='valid')(x_shortcut)
  if use_bn:
    x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)
  # Final Path.
  x = tf.keras.layers.Add()([x, x_shortcut])
  x = tf.keras.layers.Activation(activation)(x)
  if use_dropout:
    x = tf.keras.layers.Dropout(dropout_rate)(x)
  return x


def __identity_block(x, filter_start=64, kernel_size=(2, 2),
                     use_bn=True, activation='relu'):
  # resnet identity block.
  x_shortcut = x # (32, 32, 3)
  # Path 1
  x = tf.keras.layers.Conv2D(filters=filter_start*1, kernel_size=(1, 1),
                            strides=(1, 1), padding='valid')(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  # Path 2
  x = tf.keras.layers.Conv2D(filters=filter_start*2, kernel_size=kernel_size,
                            strides=(1, 1), padding='same')(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  # Path 3
  x = tf.keras.layers.Conv2D(filters=filter_start*4, kernel_size=(1, 1),
                            strides=(1, 1), padding='valid')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # Final Path.
  x = tf.keras.layers.Add()([x, x_shortcut])
  x = tf.keras.layers.Activation(activation)(x)
  return x

def __depthwise_block(x, filters, strides, alpha=1.0,
                      use_bn=True, use_dropout=False, 
                      dropout_rate=0.25, activation='relu'):
  
  # depthwise convolution.
  filters = int(filters * alpha)

  # Depthwise
  x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                      strides=strides,
                      use_bias=False,
                      padding='same')(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)

  # Pointwise
  x = tf.keras.layers.Conv2D(filters,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             use_bias=False,
                             padding='same')(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(activation)(x)
  if use_dropout:
    x = tf.keras.layers.Dropout(dropout_rate)(x)
  return x
