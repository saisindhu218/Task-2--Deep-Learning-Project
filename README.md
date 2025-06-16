# Task-2--Deep-Learning-Project

*Company*   : CODTECH IT SOLUTIONS

*Name*      : Rachabattuni Sai Sindhu

*Intern ID* : CT06DG1263

*Domain*    : Data Science

*Duration*  : 6 Weeks

*Mentor*    : Neela Santosh


            Hello! This is my write-up for Task 2 of the CODTECH Data Science Internship. The goal of this task was to build a deep learning model that could classify images — and trust me, it was such an exciting and hands-on experience. As an MCA student, diving into real-world deep learning was both a challenge and a big learning opportunity. The project focused on image classification using the Fashion MNIST dataset. I built the model using Python and PyTorch, and I worked entirely in Visual Studio Code (VS Code), which is the IDE I’m most comfortable with. To make sure everything was clean and organized, I created a dedicated Python virtual environment called datascience_env, where I installed all the necessary libraries.

### Tools and Technologies I Used
Here’s everything I used to make this project happen:
* Python for all the coding.
* PyTorch for building and training the deep learning model.
* torchvision for loading the dataset and applying image transformations.
* NumPy for numerical operations and conversions.
* Scikit-learn for generating classification reports and confusion matrices.
* Matplotlib and Seaborn for plotting the training results and visualizing the model’s performance.
* VS Code as my coding environment.
* pip for installing all required packages inside my virtual environment.

### Dataset: Fashion MNIST
The dataset I worked with is Fashion MNIST. It’s a collection of grayscale images (28x28 pixels) of different clothing items like T-shirts, trousers, dresses, and sneakers. There are 10 classes in total, and each image is labeled according to the clothing category it belongs to.
The dataset includes:
* 60,000 training images
* 10,000 testing images
It was perfect for this task because it's widely used for practicing deep learning in image classification.

### Project Workflow
#### Step 1: Data Preprocessing
I used torchvision to load the dataset. Each image was converted into a tensor and normalized to ensure better training performance. Then, I used DataLoader to manage data batching and shuffling efficiently during both training and testing.

#### Step 2: Building the CNN
I created a convolutional neural network (CNN) called SimpleCNN. It had:
* Two convolutional layers with ReLU activation and max pooling
* A dropout layer to reduce overfitting
* A couple of fully connected layers for final classification
This setup helped the model extract meaningful features from the images before making predictions.

#### Step 3: Training the Model
I trained the model for 10 epochs. For each epoch, I:
* Performed a forward pass to predict outputs
* Calculated the loss using CrossEntropyLoss
* Applied backpropagation
* Updated weights using the Adam optimizer
I also kept track of the training and validation loss and accuracy after every epoch to see how well the model was learning.

#### Step 4: Evaluating the Model
Once the training was complete, I tested the model on the test dataset. I generated a classification report to check precision, recall, and F1-score for each class. Then, I created a confusion matrix and visualized it using Seaborn to see which classes the model confused most often.

#### Step 5: Visualizing the Results
I plotted:
* Training and validation loss/accuracy curves to monitor learning behavior
* A grid of 25 test images showing actual and predicted labels — green text if correct, red if wrong — to get a visual sense of how well the model was doing

#### Step 6: Saving the Model
After everything, I saved the trained model to a file named my\_fashion\_mnist\_cnn.pth. This way, I can reuse the model in the future without needing to retrain it.

### Final Thoughts
Working on this deep learning project was a real confidence booster. From loading the data to building and evaluating a model, every step taught me something new. It wasn’t just a coding task — it felt like building something intelligent with real-life applications.
I’m really grateful to have had the chance to do this as part of my internship. This task made me even more excited about learning machine learning and artificial intelligence in greater depth. I hope to build even more advanced models in the future using what I’ve learned here.

### Key Learnings from This Task
This task really helped me level up my understanding of deep learning. Some of the biggest things I learned include:
* How to go from raw image data to a trained classification model.
* Building CNNs and understanding how convolution, pooling, and dense layers work together.
* Working with PyTorch more confidently — from tensor operations to writing training loops.
* The importance of tracking both training and validation metrics to detect overfitting or underfitting.
* How to evaluate models using metrics beyond just accuracy — like precision and recall — and how to interpret a confusion matrix.
* Why visualizations are so important in deep learning projects — they made everything easier to understand.
* Managing a project using virtual environments helped me keep things organized and avoid library conflicts.
* Troubleshooting errors like missing imports, incorrect shapes, or missing libraries gave me more confidence in debugging PyTorch code.
* Lastly, I understood the value of saving models and making projects reproducible.

