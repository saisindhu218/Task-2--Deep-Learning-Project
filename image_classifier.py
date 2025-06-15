# image_classifier.py
# This script implements a Deep Learning Model for Image Classification (Fashion MNIST)
# using PyTorch. It includes data loading, model definition, training, evaluation,
# visualization of results, and model saving.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting PyTorch Image Classification Project ---")

# --- Step 2: Choose and Load Your Dataset (Fashion MNIST) ---
print("\nLoading Fashion MNIST Dataset...")

num_classes = 10 # Fashion MNIST has 10 classes (0-9)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training dataset size: {len(train_dataset)} images")
print(f"Test dataset size: {len(test_dataset)} images")
print(f"Number of training batches (batch_size={batch_size}): {len(train_loader)}")
print(f"Number of test batches (batch_size={batch_size}): {len(test_loader)}")

try:
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Shape of a sample image batch: {sample_images.shape}")
    print(f"Shape of a sample label batch: {sample_labels.shape}")
except Exception as e:
    print(f"Could not inspect sample batch: {e}")
    print("This might happen if the dataset download failed or is incomplete.")


# --- Step 4: Design Your Deep Learning Model (CNN) ---
print("\nDefining the CNN Model Architecture...")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 128 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=num_classes)

print("\nModel Architecture:")
print(model)

print("\n--- Model Definition Complete. Ready for Training! ---")

# --- Step 5: Compiling and Training the Model ---
print("\n--- Step 5: Compiling and Training the Model ---")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# --- Initialize lists to store training history for plotting ---
train_losses_history = []
val_losses_history = []
train_accuracies_history = []
val_accuracies_history = []

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses_history.append(avg_train_loss)
    train_accuracies_history.append(train_accuracy)

    model.eval()
    validation_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs_val = model(images)
            loss_val = criterion(outputs_val, labels)
            validation_loss += loss_val.item()

            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    avg_val_loss = validation_loss / len(test_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses_history.append(avg_val_loss)
    val_accuracies_history.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_accuracy:.2f}%')

print("\n--- Model Training Complete! ---")


# --- Step 6: Evaluating the Model ---
print("\n--- Step 6: Evaluating the Model ---")

model.eval()

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

print("\nClassification Report:")
print(classification_report(all_true_labels, all_predictions, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_true_labels, all_predictions)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\n--- Model Evaluation Complete! ---")


# --- Step 7: Visualizing Results ---
print("\n--- Step 7: Visualizing Results ---")

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies_history, label='Training Accuracy')
plt.plot(val_accuracies_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(train_losses_history, label='Training Loss')
plt.plot(val_losses_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize some predictions (IMPROVED VERSION)
print("\nVisualizing some test predictions (improved clarity)...")
# Get a batch of test images and labels
dataiter = iter(test_loader)
images, labels = next(dataiter) # Get a new batch for visualization

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Display images with predictions - IMPROVED
fig = plt.figure(figsize=(15, 15)) # Increased overall figure size
fig.suptitle('Sample Predictions: True vs. Predicted Labels', fontsize=16) # Add a main title

# You can adjust num_display_images for fewer or more images if needed
num_display_images = 25 # Display 25 images (5x5 grid)
if num_display_images > len(images):
    num_display_images = len(images) # Don't try to display more than available

for idx in np.arange(num_display_images):
    ax = fig.add_subplot(5, 5, idx + 1, xticks=[], yticks=[]) # 5x5 grid

    # Unnormalize image for display: revert from [-1, 1] to [0, 1]
    img = images[idx] / 2 + 0.5
    img = img.squeeze() # Remove the channel dimension for grayscale (1, 28, 28) -> (28, 28)

    plt.imshow(img.cpu().numpy(), cmap='gray') # Move to CPU and convert to NumPy

    # Determine title text and color
    true_label_name = class_names[labels[idx].item()]
    predicted_label_name = class_names[predicted[idx].item()]
    is_correct = (predicted[idx].item() == labels[idx].item())
    
    color = 'green' if is_correct else 'red'
    title = f"True: {true_label_name}\nPred: {predicted_label_name}"

    # Set title with background for clarity
    ax.set_title(title, color=color, fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)) # Added white background to text

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()

print("\n--- Results Visualized! ---")


# --- Step 8: Saving the Model ---
print("\n--- Step 8: Saving the Model ---")

model_save_path = './my_fashion_mnist_cnn.pth'
torch.save(model.state_dict(), model_save_path)

print(f"Model weights saved successfully to: {model_save_path}")

print("\n--- Project Complete! ---")