# TF CUSTOM LAYERS

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
  return x

fft_layer = tf.keras.layers.Lambda(image_fft)
