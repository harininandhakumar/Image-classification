# Image-classification

Name : HARINI N

Intern ID: CT04DL783

Domain : MACHINE LEARNING

Duration: 4 weeks

Mentor : NEELA SANTOSH

DESCRIPTION:

The code presented is a complete workflow for training, evaluating, and visualizing the performance of a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow and Keras libraries in Python. The CIFAR-10 dataset is a well-known benchmark in the field of computer vision, consisting of 60,000 32x32 color images divided into 10 different classes, such as airplanes, cars, birds, cats, and so on. This workflow consists of the following main stages: data preprocessing, model building, model training, evaluation, and performance visualization.

1. Data Loading and Preprocessing
The code begins by loading the CIFAR-10 dataset using tf.keras.datasets.cifar10.load_data(). This dataset is divided into two parts: 50,000 images for training and 10,000 images for testing. The image data is then normalized by dividing the pixel values by 255.0. This transforms the original pixel values from the range [0, 255] to [0, 1], which helps the model converge faster and more reliably during training. The labels are flattened using flatten() to convert them from a 2D array to a 1D array for compatibility with certain Keras methods.

2. Model Construction
A sequential CNN model is defined using Keras' Sequential API. The model architecture is fairly typical for image classification tasks:
A Conv2D layer with 32 filters and a 3x3 kernel is used first, followed by a MaxPooling2D layer to reduce the spatial dimensions.
This is followed by another pair of Conv2D and MaxPooling2D layers with 64 filters.
An additional Conv2D layer with 64 filters is added to increase the depth.
The output of the last convolutional layer is flattened into a 1D vector.
A fully connected (Dense) layer with 64 neurons and ReLU activation processes this vector.
Finally, a Dense output layer with 10 units and softmax activation is used to classify images into one of the 10 categories.

3. Model Training
The model is trained for 10 epochs with a batch size of 64 using the fit() method. A validation split of 10% is used, which means 10% of the training data is held out for validation during each epoch. This helps monitor the model’s ability to generalize to unseen data and can be useful for early stopping or tuning.

4. Model Evaluation
After training, the model is evaluated on the test set using model.evaluate(). This provides the final test accuracy and loss, which are important metrics to gauge the effectiveness of the model.

5. Prediction and Metrics
The trained model predicts the class probabilities for each test image using model.predict(). The class with the highest probability is selected using np.argmax(). The predicted labels are then compared to the true labels using classification_report() and confusion_matrix() from the sklearn.metrics module. The classification report provides detailed metrics such as precision, recall, and F1-score for each class. The confusion matrix shows how many predictions were made correctly or incorrectly for each class, which helps identify class-wise performance.

6. Visualization
Finally, the training and validation accuracies stored in the history object are plotted over the epochs using matplotlib.pyplot. This visual representation helps understand whether the model is overfitting, underfitting, or learning well. A divergence between training and validation curves might indicate overfitting.


OUTPUT:

![Image](https://github.com/user-attachments/assets/9877d7f4-b3b7-4a15-9ea3-dc33c03f9dfd)
![Image](https://github.com/user-attachments/assets/8f928c8b-8e60-4679-bc9a-c826629ce457)
![Image](https://github.com/user-attachments/assets/b7f425f2-f340-42ee-8275-5413dbd14962)
![Image](https://github.com/user-attachments/assets/d7d5ac4d-344e-4e48-9bfd-f02fe8875e69)
![Image](https://github.com/user-attachments/assets/3936ea84-869f-4079-8f41-329ceddfd96f)
![Image](https://github.com/user-attachments/assets/cffe51b7-57fb-4129-a68e-4b4447100f78)
