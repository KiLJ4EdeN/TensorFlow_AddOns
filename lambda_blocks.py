# TF CUSTOM LAYERS
import tensorflow as tf
import cv2


def image_to_gray(x):
  return tf.image.rgb_to_grayscale(tf.cast(x, dtype=tf.float32))

gray_layer = tf.keras.layers.Lambda(image_to_gray)

def image_gradient(x):
  # make sure dtype is correct.
  x = tf.cast(x, dtype=tf.float32)
  x = tf.image.rgb_to_grayscale(x)
  dx, dy = tf.image.image_gradients(x)
  return dx + dy
  
gradient_layer = tf.keras.layers.Lambda(image_gradient)

def image_to_hsv(x):
  # make sure dtype is correct.
  x = tf.cast(x, dtype=tf.float32)
  x = tf.image.rgb_to_hsv(x)
  return x

hsv_layer = tf.keras.layers.Lambda(image_to_hsv)

def sobel_gradient(x):
  x = tf.cast(x, dtype=tf.float32)
  x = tf.image.rgb_to_grayscale(x)
  x = tf.image.sobel_edges(x)
  # add dx and dy optionally.
  return x[:, :, :, :, 0] + x[:, :, :, :, 1]

sobel_layer = tf.keras.layers.Lambda(sobel_gradient)

def image_fft(x):
  x = tf.cast(x, dtype=tf.float32)
  x = tf.image.rgb_to_grayscale(x)
  x = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))
  return tf.cast(tf.abs(x), dtype=tf.float32)

fft_layer = tf.keras.layers.Lambda(image_fft)


def gabor_filter(x):
  x = tf.cast(x, dtype=tf.float32)
  x = tf.image.rgb_to_grayscale(x)
  params = {'ksize':(3, 3), 'sigma':1.0, 'theta': 0, 'lambd':5.0, 'gamma':0.02}
  kernel = cv2.getGaborKernel(**params)
  kernel = tf.expand_dims(kernel, 2)
  kernel = tf.expand_dims(kernel, 3)
  return tf.nn.conv2d(image, filter, strides=[1, 1, 1, 1], padding='SAME')

gabor_layer = tf.keras.layers.Lambda(gabor_filter)
