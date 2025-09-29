import numpy as np
import pandas as pd
import time
import os
from network import NeuralNetwork

# Load data directly
print("Loading data...")
train_df = pd.read_csv("data/MNIST_CSV/mnist_train.csv")
test_df = pd.read_csv("data/MNIST_CSV/mnist_test.csv")

# Extract and normalize data
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values.T / 255.0  # Transpose for (784, m) format
y_test = test_df.iloc[:, 0].values  
X_test = test_df.iloc[:, 1:].values.T / 255.0  # Transpose for (784, m) format

def one_hot(y):
    return np.eye(10)[y].T

# One-hot encode labels
y_train_one_hot = one_hot(y_train)
y_test_one_hot = one_hot(y_test)

# Initialize neural network
hidden_size = 128
learning_rate = 0.1
nn = NeuralNetwork(hidden_size)

# Training parameters
epochs = 10
batch_size = 64
m = X_train.shape[1]
num_batches = m // batch_size

# Track best model
best_accuracy = 0
best_model = None

# Training loop
for epoch in range(epochs):
    epoch_start_time = time.time()
    
    # Shuffle data
    permutation = np.random.permutation(m)
    X_train_shuffled = X_train[:, permutation]
    y_train_shuffled = y_train_one_hot[:, permutation]
    
    epoch_train_loss = 0
    epoch_train_acc = 0
    
    # Batch training
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, m)
        
        X_batch = X_train_shuffled[:, start_idx:end_idx]
        y_batch = y_train_shuffled[:, start_idx:end_idx]
        
        # Forward pass
        Z1, A1, Z2, A2 = nn.forward(X_batch)
        
        # Calculate loss and accuracy for this batch
        batch_loss = nn.get_loss(A2, y_batch.argmax(axis=0))
        batch_predictions = nn.get_predictions(A2)
        batch_accuracy = nn.get_accuracy(batch_predictions, y_batch.argmax(axis=0))
        
        # Backward pass
        dW1, db1, dW2, db2 = nn.backward(X_batch, y_batch, Z1, A1, Z2, A2)
        
        # Update parameters
        nn.update_param(dW1, db1, dW2, db2, learning_rate)
        
        # Accumulate metrics
        epoch_train_loss += batch_loss
        epoch_train_acc += batch_accuracy
        
        # Print batch progress every 200 batches
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{num_batches} - Loss: {batch_loss:.4f} - Acc: {batch_accuracy*100:.2f}%")

    # Calculate average metrics for the epoch
    avg_train_loss = epoch_train_loss / num_batches
    avg_train_acc = epoch_train_acc / num_batches

    # Test on validation set
    _, _, _, A2_test = nn.forward(X_test)
    test_loss = nn.get_loss(A2_test, y_test)
    test_predictions = nn.get_predictions(A2_test)
    test_accuracy = nn.get_accuracy(test_predictions, y_test)
    
    epoch_time = time.time() - epoch_start_time
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc*100:.2f}% - Test Loss: {test_loss:.4f} - Test Acc: {test_accuracy*100:.2f}% - Time: {epoch_time:.1f}s")


    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = {
            'W1': nn.W1.copy(),
            'b1': nn.b1.copy(),
            'W2': nn.W2.copy(),
            'b2': nn.b2.copy()
        }
        print(f"→ New best model saved! (Accuracy: {test_accuracy*100:.2f}%)")

print(f"Training completed! Best accuracy: {best_accuracy*100:.2f}%")