# Importing tensorflow to load the saved model
import tensorflow as tf

# Importing numpy for data manipulation
import numpy as np

# Importing image to deal with and manipulate images
from keras.preprocessing import image

# Loading the saved model
model = tf.keras.models.load_model('./')

# Getting the path of the image
path = input("Enter the Path of the Image: ")

# Loading the image from the path taken from the user
image_to_predict= image.load_img(path, target_size = (256, 256))

# Coverting the image to array
image_to_predict = image.img_to_array(image_to_predict)
image_to_predict = np.expand_dims(image_to_predict, axis = 0)

# Rescaling the image so that the model can take this as an input
image_to_predict = image_to_predict / 255.0

# Getting the prediction from the model
result = model.predict(image_to_predict)

# Printing out the result in a user friendly
if result[0][0]>0.5:
    print('The image appears Normal')
elif result[0][0] < 0.5:
    print('The image appears Cancerous')