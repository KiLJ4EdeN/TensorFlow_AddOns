from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

def get_effnet(included_layers=1):
  inputs = Input(shape=(224, 224, 3))
  model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                              input_shape=(224, 224, 3), pooling='avg')
  # reduce trainability     
  for layer in model.layers[:-included_layers]:
    layer.trainable = False
  print(f'Training {included_layers} Layers from {len(model.layers)}...') 
  # add classification layers
  last_layer = model(inputs)
  x = Dense(1280, activation='relu')(last_layer)
  x = Dense(2048, activation='relu')(x)
  out = Dense(1, activation='sigmoid')(x)
  custom_model = tf.keras.models.Model(inputs=inputs, outputs=out)
  print(custom_model.summary())
  return model

model = get_effnet(included_layers=-1)
