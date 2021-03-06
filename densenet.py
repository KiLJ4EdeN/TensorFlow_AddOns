import tensorflow as tf


def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.1,
             dropout_rate=0.1, weight_decay=1e-4, classes=7, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    '''
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')
    '''
    concat_axis = 3
    img_input = tf.keras.layers.Input(shape=(64, 64, 1), name='data')
    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121

    # Initial convolution
    x = tf.keras.layers.ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = tf.keras.layers.Conv2D(nb_filter, kernel_size=(7, 7), name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    # x = tf.keras.layers.Scale(axis=concat_axis, name='conv1_scale')(x)
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    # x = tf.keras.layers.Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = tf.keras.layers.Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x = tf.keras.layers.Dense(classes, name='fc6')(x)
    x = tf.keras.layers.Activation('softmax', name='prob')(x)

    model = tf.keras.models.Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    # x = tf.keras.layers.Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base+'_x1')(x)
    x = tf.keras.layers.Convolution2D(inter_channel, kernel_size=(1, 1), name=conv_name_base+'_x1')(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    # x = tf.keras.layers.Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base+'_x2')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = tf.keras.layers.Conv2D(nb_filter, kernel_size=(3, 3), name=conv_name_base+'_x2')(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    # x = tf.keras.layers.Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base)(x)
    x = tf.keras.layers.Conv2D(int(nb_filter * compression), kernel_size=(1, 1), name=conv_name_base)(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = tf.keras.layers.concatenate([concat_feat, x], name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
    
    
if __name__ == '__main__':
  # create a densenet instance.
  model = DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5,
             dropout_rate=0.5, weight_decay=1e-4, classes=7, weights_path=None)
  # model.fit()...
  
