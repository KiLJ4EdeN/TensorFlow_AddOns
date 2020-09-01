paramk = 11 # The k Parameter of Nearest Neighbors Algorithm.

# Defining KNN Graph with L0 Norm
# TODO: remove placeholders for tf2 compatibility.
x = tf.placeholder(trImages.dtype, shape=trImages.shape) # all train images, i.e., 60000 x 28 x 28
y = tf.placeholder(tImages.dtype, shape=tImages.shape[1:]) # a test image, 28 x 28

# TODO: remove contrib calls.
xThresholded = tf.clip_by_value(tf.cast(x, tf.int32), 0, 1) # x is int8 which is not supported in many tf functions, hence typecast
yThresholded = tf.clip_by_value(tf.cast(y, tf.int32), 0, 1) # clip_by_value converts dataset to tensors of 0 and 1, i.e., 1 where tensor is non-zero
computeL0Dist = tf.count_nonzero(xThresholded - yThresholded, axis=[1,2]) # Computing L0 Norm by reducing along axes
findKClosestTrImages = tf.contrib.framework.argsort(computeL0Dist, direction='ASCENDING') # sorting (image) indices in order of ascending metrics, pick first k in the next step
findLabelsKClosestTrImages = tf.gather(trLabels, findKClosestTrImages[0:paramk]) # doing trLabels[findKClosestTrImages[0:paramk]] throws error, hence this workaround
findULabels, findIdex, findCounts = tf.unique_with_counts(findLabelsKClosestTrImages) # examine labels of k closest Train images
findPredictedLabel = tf.gather(findULabels, tf.argmax(findCounts)) # assign label to test image based on most occurring labels among k closest Train images

# Let's run the graph
numErrs = 0
numTestImages = np.shape(tLabels)[0]
numTrainImages = np.shape(trLabels)[0] # so many train images

# remove session.
with tf.Session() as sess:
  for iTeI in range(0,numTestImages): # iterate each image in test set
    predictedLabel = sess.run([findPredictedLabel], feed_dict={x:trImages, y:tImages[iTeI]})   

    if predictedLabel == tLabels[iTeI]:
      numErrs += 1
      print(numErrs,"/",iTeI)
      print("\t\t", predictedLabel[0], "\t\t\t\t", tLabels[iTeI])
      
      if (1):
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(tImages[iTeI])
        plt.title('Test Image has label %i' %(predictedLabel[0]))
        
        for i in range(numTrainImages):
          if trLabels[i] == predictedLabel:
            plt.subplot(1,2,2)
            plt.imshow(trImages[i])
            plt.title('Correctly Labeled as %i' %(tLabels[iTeI]))
            plt.draw()
            break
        plt.show()

print("# Classification Errors= ", numErrs, "% accuracy= ", 100.*(numTestImages-numErrs)/numTestImages)
      
