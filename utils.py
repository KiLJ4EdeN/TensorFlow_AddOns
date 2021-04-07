# KERAS UTILS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def train(model, X_train, X_test, Y_train, Y_test,
          batch_size, epochs, augment=True):
  
  model.compile(loss='categorical_crossentropy', 
          metrics=['accuracy'],
          optimizer='rmsprop')
  
  if augment:
    datagen = ImageDataGenerator(
                              width_shift_range=0.125,
                              height_shift_range=0.125,
                              fill_mode="constant",
                              horizontal_flip=True, cval=0.)
    datagen.fit(X_train)
    batch_size = batch_size
    epochs = epochs
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=(X_train.shape[0] // batch_size),
                        epochs=epochs, verbose=1,
                        validation_data=(X_test, Y_test),)
    
  else:
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                        verbose=1, validation_data=(X_test, Y_test))

  return model, history

def plot_history(history):
  plt.figure(figsize=(6, 2))
  plt.plot(history.history["accuracy"])
  plt.plot(history.history["val_accuracy"])
  plt.title("model accuracy")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.show()
  plt.figure(figsize=(6, 2))
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.title("model loss")       
  plt.ylabel("loss") 
  plt.xlabel("epoch")
  plt.show()
          
          
class CustomGen(keras.utils.Sequence):

    # get the filenames and the labels
    def __init__(self, image_filenames, labels, batch_size, size=(128, 128, 3)):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.size = size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([
            resize(imread(str(file_name)), self.size)
            for file_name in batch_x]) / 255.0, np.array(batch_y)
