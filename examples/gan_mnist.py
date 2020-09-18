from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Reshape

class MNIST():
  def __init__(self):
    self.load_dataset()
    self.load_real_samples()

  # load the mnist dataset into disk.
  def load_dataset(self):
    (X_train, Y_train), (X_test, Y_test) = load_data()
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  
  # load the real sample source.
  def load_real_samples(self):
    X = np.expand_dims(self.X_train, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    self.X_real = X

  def plot_samples(self):
    for i in range(25):
      plt.subplot(5, 5, i+1)
      plt.axis('off')
      plt.imshow(self.X_train[i], cmap='gray_r')
    plt.show()
    
# mnist = MNIST()
# mnist.plot_samples()

class GAN():
  def __init__(self):
    self.dataset = MNIST()
    self.discriminator = self.define_discriminator()
    self.generator = self.define_generator()
    self.composite = self.define_gan()

  @staticmethod
  def generate_latent_points(latent_dim, n_samples):
    x = np.random.randn(latent_dim * n_samples)
    return x.reshape(n_samples, latent_dim)

  # get a batch of real samples.
  def generate_real_samples(self, n_samples):
    ix = np.random.randint(0, self.dataset.X_real.shape[0], n_samples)
    X = self.dataset.X_real[ix]
    y = np.ones((n_samples, 1))
    return X, y

  # get a batch of fake samples.
  def generate_fake_samples(self, latent_dim, n_samples):
    x = self.generate_latent_points(latent_dim, n_samples)
    X = self.generator.predict(x)
    y = np.zeros((n_samples, 1))
    return X, y

  @staticmethod
  def define_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))   

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

  @staticmethod 
  def define_generator(latent_dim=100):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # 14*14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # 28*28
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model

  def define_gan(self):
    self.discriminator.trainable = False

    model = Sequential()
    model.add(self.generator)
    model.add(self.discriminator)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

  def train_discriminator(self, n_iter=100, n_batch=256):
    half_batch = int(n_batch / 2)
    for i in range(n_iter):
      X_real, Y_real = self.generate_real_samples(half_batch)
      _, real_acc = self.discriminator.train_on_batch(X_real, Y_real)

      X_fake, Y_fake = self.generate_fake_samples(half_batch)
      _, fake_acc = self.discriminator.train_on_batch(X_fake, Y_fake)

      print('>%d real%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

  def train_gan(self, latent_dim=100, n_epochs=100, n_batch=256):
    for i in range(n_epochs):
      x_gan = self.generate_latent_points(latent_dim, n_batch)
      # treat fakes as real.
      y_gan = np.ones((n_batch, 1))
      gan_model.train_on_batch(x_gan, y_gan)

  
  def train(self, latent_dim=100, n_epochs=100, n_batch=256):
    bat_per_epo = int(self.dataset.X_real.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
    # enumerate batches over the training set
      for j in range(bat_per_epo):
        # get randomly selected 'real' samples
        X_real, y_real = self.generate_real_samples(half_batch)
        # generate 'fake' examples
        X_fake, y_fake = self.generate_fake_samples(latent_dim, half_batch)
        # create training set for the discriminator
        X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
        # update discriminator model weights
        d_loss, _ = self.discriminator.train_on_batch(X, y)
        # prepare points in latent space as input for the generator
        X_gan = self.generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = self.composite.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))

  
  # evaluate the discriminator, plot generated images, save generator model
  def summarize_performance(self, epoch, latent_dim=100, n_samples=100):
    # prepare real samples
    X_real, y_real = self.generate_real_samples(n_samples)
    # evaluate discriminator on real examples
    _, acc_real = self.discriminator.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = self.generate_fake_samples(latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    self.generator.save(filename)

  @staticmethod
  # create and save a plot of generated images (reversed grayscale)
  def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
      # define subplot
      plt.subplot(n, n, 1 + i)
      # turn off axis
      plt.axis('off')
      # plot raw pixel data
      plt.imshow(examples[i, :, :, 0], cmap='gray_r')
      # save plot to file
      filename = 'generated_plot_e%03d.png' % (epoch+1)
      plt.savefig(filename)
      plt.close()
      
if __name__ == '__main__':
  gan = GAN()
  print(gan.discriminator.summary())
  print(gan.generator.summary())
  print(gan.composite.summary())
  gan.train(latent_dim=100, n_epochs=100, n_batch=256)
