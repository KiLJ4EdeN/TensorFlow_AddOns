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
