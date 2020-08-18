# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.001
training_steps = 1000
batch_size = 256
display_step = 100

# Network parameters.
n_hidden_1 = 128 # 1st layer number of neurons.
n_hidden_2 = 256 # 2nd layer number of neurons.

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Convert to float32.
X_train = tf.Variable(X_train, dtype=tf.float32)
X_test = tf.Variable(X_test, dtype=tf.float32)
# Flatten images to 1-D vector of 784 features (28*28).
X_train = tf.reshape(X_train, [-1, num_features])
X_test = tf.reshape(X_test, [-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
X_train = X_train / 255.
X_test = X_test / 255.

print(X_train.shape)
print(X_test.shape)

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
# repeat adds the data again, prefetch speeds up outputs with the cost of ram.
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

num_hidden_units = [n_hidden_1, n_hidden_2, num_classes]
random_normal = tf.initializers.RandomNormal()
# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W1 = tf.Variable(random_normal([num_features, num_hidden_units[0]]), name="weight1")
W2 = tf.Variable(random_normal([num_hidden_units[0], num_hidden_units[1]]), name="weight2")
W3 = tf.Variable(random_normal([num_hidden_units[1], num_hidden_units[2]]), name="weight3")
# Bias of shape [10], the total number of classes.
b1 = tf.Variable(tf.zeros([num_hidden_units[0]]), name="bias1")
b2 = tf.Variable(tf.zeros([num_hidden_units[1]]), name="bias2")
b3 = tf.Variable(tf.zeros([num_hidden_units[2]]), name="bias3")

def multilayer_perceptron(x):
    # Apply softmax to normalize the logits to a probability distribution.
    h1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
    return tf.nn.softmax(h3)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = multilayer_perceptron(x)
        loss = cross_entropy(pred, y)

    # Compute gradients.
    gradients = g.gradient(loss, [W1, W2, W3, b1, b2, b3])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W1, W2, W3, b1, b2, b3]))

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = multilayer_perceptron(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set.
pred = multilayer_perceptron(X_test)
print("Test Accuracy: %f" % accuracy(pred, Y_test))

# Visualize predictions.
# Predict 5 images from validation set.
n_images = 5
test_images = X_test[:n_images]
predictions = multilayer_perceptron(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(tf.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % tf.argmax(predictions.numpy()[i]))
