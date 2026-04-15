# Deep-Learning-Project


###  
Hello! This is my write-up for Task 2 of the CODTECH Data Science Internship. The goal of this task was to build a deep learning model that could classify images and trust me, it was such an exciting and hands-on experience. As an MCA student, diving into real-world deep learning was both a challenge and a big learning opportunity. The project focused on image classification using the Fashion MNIST dataset. I built the model using Python and PyTorch, and I worked entirely in Visual Studio Code (VS Code), which is the IDE I’m most comfortable with. To make sure everything was clean and organized, I created a dedicated Python virtual environment called datascience_env, where I installed all the necessary libraries.

## Tools and Technologies I Used
Here’s everything I used to make this project happen:
* Python for all the coding.
* PyTorch for building and training the deep learning model.
* torchvision for loading the dataset and applying image transformations.
* NumPy for numerical operations and conversions.
* Scikit-learn for generating classification reports and confusion matrices.
* Matplotlib and Seaborn for plotting the training results and visualizing the model’s performance.
* VS Code as my coding environment.
* pip for installing all required packages inside my virtual environment.

## Dataset: Fashion MNIST
The dataset I worked with is Fashion MNIST. It’s a collection of grayscale images (28x28 pixels) of different clothing items like T-shirts, trousers, dresses, and sneakers. There are 10 classes in total, and each image is labeled according to the clothing category it belongs to.
The dataset includes:
* 60,000 training images
* 10,000 testing images
It was perfect for this task because it's widely used for practicing deep learning in image classification.

## Project Workflow
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
* A grid of 25 test images showing actual and predicted labels — green text if correct, red if wrong to get a visual sense of how well the model was doing

#### Step 6: Saving the Model
After everything, I saved the trained model to a file named my\_fashion\_mnist\_cnn.pth. This way, I can reuse the model in the future without needing to retrain it.

## Key Learnings from This Task
This task really helped me level up my understanding of deep learning. Some of the biggest things I learned include:
* How to go from raw image data to a trained classification model.
* Building CNNs and understanding how convolution, pooling, and dense layers work together.
* Working with PyTorch more confidently from tensor operations to writing training loops.
* The importance of tracking both training and validation metrics to detect overfitting or underfitting.
* How to evaluate models using metrics beyond just accuracy like precision and recall and how to interpret a confusion matrix.
* Why visualizations are so important in deep learning projects they made everything easier to understand.
* Managing a project using virtual environments helped me keep things organized and avoid library conflicts.
* Troubleshooting errors like missing imports, incorrect shapes, or missing libraries gave me more confidence in debugging PyTorch code.
* Lastly, I understood the value of saving models and making projects reproducible.

## Final Thoughts

As a beginner stepping into the world of deep learning, this project was a turning point for me. It was my first time building and training a complete neural network from scratch and that too on image data, which felt both exciting and intimidating at the start. But as I progressed through each phase of this project, from loading and preprocessing the Fashion MNIST dataset to building a CNN model and evaluating its performance, I realized that deep learning is not just about complex math or algorithms it's about teaching machines to recognize patterns in a way that mimics human thinking.
This task boosted my confidence and helped me appreciate the power of PyTorch and how modular and flexible it is for building deep learning models. I also understood the importance of concepts like overfitting, model evaluation metrics beyond accuracy, and why visualizing learning curves can make a big difference in understanding model behavior. It taught me how every line of code whether defining layers, applying activations, or computing loss plays a crucial role in making the model smarter.

More than anything, it gave me a sense of achievement. It showed me that even as a student who’s still learning, I can build intelligent systems that can make decisions, learn from data, and improve over time. I now feel more equipped to explore advanced topics like transfer learning, custom datasets, and deploying models in real-world scenarios.
This project wasn't just about submitting a task it was about discovering what I’m capable of. It’s motivated me to dive deeper into machine learning and AI, and I’m excited to continue this journey, one model at a time.


## OUTPUT:

### This is for model evaluation:
![Image](https://github.com/user-attachments/assets/684a5b07-a1b5-434a-aac4-e02538d2eed1)

![Image](https://github.com/user-attachments/assets/70fdec0f-30e9-4e2c-b6d2-8b824113864a)

### Visualizing the Results:
![Image](https://github.com/user-attachments/assets/fd105408-b414-4c46-a402-e2f347f47aa8)

![Image](https://github.com/user-attachments/assets/480a147a-08ed-41a0-921a-b87673b92216)

And saved the model as my_fashion_mnist_cnn.pth format.


### Rachabattuni Sai Sindhu
