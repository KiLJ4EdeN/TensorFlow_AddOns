import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Add, BatchNormalization, concatenate
from tensorflow.keras.layers import AveragePooling2D


def Transfer_Learn(input_shape, classes, included_layers=1, model='ResNet50'):
  model_database = ['VGG16', 'ResNet50', 'ResNet50V2', 'DenseNet121', 'DenseNet169',
                    'DenseNet201', 'Xception', 'MobileNet', 'MobileNetV2', 
                    'NASNetMobile', 'InceptionV3', 'EfficientNetB0']
  image_input = Input(input_shape)


  print('Model is being loaded...')
  # select the model
  if model == 'ResNet50':
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg')  
  elif model == 'ResNet50V2':
    model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg')       
  elif model == 'VGG16':
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg')    
  elif model == 'DenseNet121':
    model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg')   
  elif model == 'DenseNet169':
    model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'DenseNet201':
    model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'Xception':
    model = tf.keras.applications.Xception(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'MobileNet':
    model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'MobileNetV2':
    model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'InceptionV3':
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'NASNetMobile':
    model = tf.keras.applications.NASNetMobile(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  elif model == 'EfficientNetB0':
    model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
    input_shape=input_shape, pooling='avg') 
  else:
    print(f'Wrong model name. \nAvailable models are: {model_database}')



  # reduce trainability     
  for layer in model.layers[:-included_layers]:
    layer.trainable = False
  print(f'Training {included_layers} Layers from {len(model.layers)}...') 
  # add classification layers
  last_layer = model(image_input)
  if classes > 1:
    act = 'softmax'
  else:
    act = 'sigmoid'
  out = Dense(classes, activation=act, name='output_layer')(last_layer)
  custom_model = tf.keras.models.Model(inputs=image_input,outputs=out)

  return custom_model
