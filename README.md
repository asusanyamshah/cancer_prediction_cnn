# Cancer Prediction Using Neural Network

### About the Dataset

The dataset includes breast ultrasound images. There are two types of images, one having cancer, and another without cancer. All the images have been split into training data and testing data. Training data will be used to train the model, and the testing data to test the model. 

These images were collected from:

    1. Kaggle: Some of the training and testing data was collected from this site: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset. 

    2. A Local Hospital

The sources are trusted and valid data is used. 

Inside the ```train``` folder, we have two more folders named ```cancer``` and ```normal```. The ```cancer``` folder contains images of breast ultrasound showing cancer, both benign and malignant. The ```normal``` folder contains normal breast ultrasound images with no cancer. This structure is similar for the ```test``` folder. 

### Model.ipynb

A Convolutional Neural Network is made which can be trained and then used to make some predictions. Libraries named tensorflow and keras are used for making the model and image manipulation. 

The following steps were taken to make the model and train it. 

1. Data Augmentation: Data Augmentation is the generation of new data from already existing data using certain parameters to specify how the new data must be generated. This increases the size of training data and provides the Neural Network more data to work with and learn from, which increases its accuracy. 

New Data was generated using ImageDataGenerator Class, and parameters like ```shear_range```, ```zoom_range```, and ```horizontal_flip``` were used to specify the data to be created. 

2. Adding the Layers of the Neural Network. We determine the Architecture of neural network.

    1. The CNN is instantiated as s Sequential model using the command ```cnn = tf.keras.models.Sequential()```. 
    We add a 2D convolution layer. The layer contains filters, which slides across the input image, performing a mathematical operation named colvolution and gives an output. These filters are responsible for detecting specific images of an image like corners, edges, and a lot more. After the convolution operation, an activation function is applied to the output. The filters are small matrices.

    There are 3 such layers, each with the number of filters being 32, kernel_size being 3 x 3 and the activation function being applied called ```relu```. The kernel_size refers to the dimensions of the filters or matrices. 


    2. Max Pooling Layers. This layer extracts important features from the input. The MaxPool2d layer takes as input a feature map obtained from the convolution layer. The layer then performs an operation to select the maximum value in each patch of the feature map. This layer helps in dimensionality reduction and it reduces the spatial dimensions. This lead to more efficient models because it helps focus on the important features only. 

    There are 3 such layers with ```pool_size = 2``` and ```strides = 2```. The ```pool_size``` refers to a tuple of two integers which states the dimensions of the pooling windows. ```strides``` determines the number of steps used when moving the pooling window across the input. 

    3. Flatten Layer. This layer converts the multidimensional input it gets from all the layers into a one dimensional vector. There is one such layer. 

    4. Dense Layer. This layer is also known as a fully connected layer. This layer takes input from all the previous layers and is responsible for classifying the images. The ```units``` in the dense layers refers to the number of neurons present in this layer.  The ```kernel_initializer``` is ```he_normal``` refers to the way the initial weights are set. ```activation``` refers to the activation function used. This activation funciton is applied to the output of each neuron. 

    4. Dropout layer. This layer is used to prevent overfitting. It a a regularization technique, and it improves the performance and accuracy of the neural network. The Dropout Rate is 0.2. This is the probability of dropping out each neuron in a layer. 

    5. The ouput layer. The last dense layer with ```activation = 'sigmoid'``` is the output layer. 

Now the structure of the neural network is made. 

3. An optimizer is made. The optimizer named ```Adam``` is used and the learning rate is ```0.001```. The function of the optimizer is to update the weights to get better accuracy. They reduce the loss. 

4. Compiling the model. The entire model is then compiled. The optimizer is specified. The loss is calculated using the binary_crossentropy function because we are dealing with binary classification. The loss is a way to determine the accuracy of the model. The loss determines the difference between the predicted value and the true value. 

5. Early Stopping. Early Stopping is a method used to prevent overfitting and stop the training of the model when the model stopped performing well while training. Here, we monitor the ```val_loss``` or validation loss and the ```patience = 5``` states the number of epochs to wait before stopping the model if ```val_loss``` dosent improve 

5. Training the model. We then finally train the model by providing it with the training data, testing data or validation data, specifying the number of epochs and providing it the Early Stopping callback. 

6. Saving the model. After the traing process is done, we save the trained model using ```tf.keras.models.save_model(cnn, './')```. This saves the model in the current directory. We can use this model in the future without having to train it again. 

### Script.ipynb