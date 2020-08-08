# cnn_blocks
Famous cnn block implementations.

[![License](https://img.shields.io/github/license/KiLJ4EdeN/cnn_blocks)](https://img.shields.io/github/license/KiLJ4EdeN/cnn_blocks) [![Version](https://img.shields.io/github/v/tag/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/v/tag/KiLJ4EdeN/cnn_blocks) [![Code size](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/cnn_blocks)](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/cnn_blocks) [![Repo size](https://img.shields.io/github/repo-size/KiLJ4EdeN/cnn_blocks)](https://img.shields.io/github/repo-size/KiLJ4EdeN/cnn_blocks) [![Issue open](https://img.shields.io/github/issues/KiLJ4EdeN/cnn_blocks)](https://img.shields.io/github/issues/KiLJ4EdeN/cnn_blocks)
![Issue closed](https://img.shields.io/github/issues-closed/KiLJ4EdeN/cnn_blocks)


# Usage:
## Install Dependencies w pip:

1- tensorflow


## Import the block module and use the predefined  layers.
```python
from blocks import __conv_block, __dense_block, __classification_block, __parallel_block
from blocks import __depthwise_block, __indentity_block, __residual_block
```


## Examples:
### Resnet Like Architecture.
```python
import tensorflow as tf
from blocks import __identity_block, __residual_block, __dense_block, __classification_block

inputs = tf.keras.layers.Input(shape=(32, 32, 3))

x = __residual_block(inputs, filter_start=16, kernel_size=(3, 3),
                     use_bn=True, use_constraint=True,
                     use_dropout=True, constraint_rate=1,
                     dropout_rate=0.25, activation='relu')

x = __identity_block(x, filter_start=16, kernel_size=(3, 3),
                     use_bn=True, activation='relu')
x = __identity_block(x, filter_start=16, kernel_size=(3, 3),
                     use_bn=True, activation='relu')

x = __residual_block(x, filter_start=32, kernel_size=(3, 3),
                     use_bn=True, use_constraint=True,
                     use_dropout=True, constraint_rate=1,
                     dropout_rate=0.25, activation='relu')
x = __identity_block(x, filter_start=32, kernel_size=(3, 3),
                     use_bn=True, activation='relu')
x = __identity_block(x, filter_start=32, kernel_size=(3, 3),
                     use_bn=True, activation='relu')


x = __residual_block(x, filter_start=64, kernel_size=(3, 3),
                     use_bn=True, use_constraint=True,
                     use_dropout=True, constraint_rate=1,
                     dropout_rate=0.25, activation='relu')
x = __identity_block(x, filter_start=64, kernel_size=(3, 3),
                     use_bn=True, activation='relu')
x = __identity_block(x, filter_start=64, kernel_size=(3, 3),
                     use_bn=True, activation='relu')

x = __residual_block(x, filter_start=128, kernel_size=(3, 3),
                     use_bn=True, use_constraint=True,
                     use_dropout=True, constraint_rate=1,
                     dropout_rate=0.25, activation='relu')
x = __identity_block(x, filter_start=128, kernel_size=(3, 3),
                     use_bn=True, activation='relu')
x = __identity_block(x, filter_start=128, kernel_size=(3, 3),
                     use_bn=True, activation='relu')

x = __dense_block(x, unit_start=512, num_blocks=2,
                  flatten=True, use_constraint=True,
                  use_dropout=True, constraint_rate=1,
                  dropout_rate=0.25, activation='relu')

x = __classification_block(x, num_classes=100)

model = tf.keras.models.Model(inputs=inputs, outputs=x)
print(model.summary())
```

### Mobilenet Customized.
```python
import tensorflow as tf
from blocks import __depthwise_block, __dense_block, __classification_block

inputs = tf.keras.layers.Input(shape=(32, 32, 3))

x = __depthwise_block(inputs, filters=8, strides=(1, 1), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=16, strides=(2, 2), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=32, strides=(1, 1), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=64, strides=(2, 2), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=128, strides=(1, 1), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=256, strides=(2, 2), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=512, strides=(1, 1), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __depthwise_block(x, filters=1024, strides=(2, 2), alpha=1.0,
                      use_bn=True, use_dropout=True, 
                      dropout_rate=0.25, activation='relu')

x = __dense_block(x, unit_start=512, num_blocks=1,
                  flatten=True, use_constraint=True,
                  use_dropout=True, constraint_rate=1,
                  dropout_rate=0.5, activation='relu')

x = __classification_block(x, num_classes=100)

model = tf.keras.models.Model(inputs=inputs, outputs=x)
print(model.summary())
```

### Simple CNN.
```python
import tensorflow as tf
from blocks import __conv_block, __dense_block, __classification_block

# basic net.

inputs = tf.keras.layers.Input(shape=(32, 32, 3))

x = __conv_block(inputs, filter_start=64, kernel_size=(2, 2),
                 num_blocks=2,
                 use_bn=True, use_constraint=True,
                 use_dropout=True, constraint_rate=1,
                 dropout_rate=0.3, activation='relu')

x = __dense_block(x, unit_start=128, num_blocks=2,
                  flatten=True, use_constraint=True,
                  use_dropout=True, constraint_rate=1,
                  dropout_rate=0.5, activation='relu')

x = __classification_block(x, num_classes=100)

model = tf.keras.models.Model(inputs=inputs, outputs=x)
print(model.summary())
```
