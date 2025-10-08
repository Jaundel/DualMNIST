import numpy as np
import pandas as pd
import time
from pathlib import Path
import onnx
import ndonnx as ndx
from network import NeuralNetwork

# Load data (robust to current working directory)
print("Loading data...")
ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
MODELS_DIR = MODELS_ROOT / "numpy_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "mnist_numpy.onnx"
METRICS_PATH = MODELS_DIR / "best_metrics.npz"
LEGACY_MODELS_DIR = MODELS_ROOT / "numpy"
LEGACY_METRICS_PATH = LEGACY_MODELS_DIR / "best_metrics.npz"
CSV_DIR = ROOT / "data" / "MNIST_CSV"
train_df = pd.read_csv(CSV_DIR / "mnist_train.csv")
test_df = pd.read_csv(CSV_DIR / "mnist_test.csv")

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

def save_to_onnx(model_dict, path):
    """Minimal ONNX export using ndonnx."""
    W1, b1, W2, b2 = (
        model_dict['W1'].astype(np.float32),
        model_dict['b1'].astype(np.float32),
        model_dict['W2'].astype(np.float32),
        model_dict['b2'].astype(np.float32),
    )

    def predict(X):
        X = ndx.asarray(X)
        Z1 = X @ W1.T + b1.T
        A1 = ndx.maximum(0, Z1)
        Z2 = A1 @ W2.T + b2.T
        # Z2 is a single sample vector, so take argmax over axis 0
        return ndx.argmax(Z2, axis=0)

    X = ndx.argument(shape=(784,), dtype=ndx.float32)
    y = predict(X)
    onnx_model = ndx.build({"X": X}, {"y": y})
    onnx.save(onnx_model, str(path))

# Training parameters
epochs = 10
batch_size = 64
m = X_train.shape[1]
num_batches = m // batch_size

# Track best model (persisted globally)
stored_metrics = None
if METRICS_PATH.exists():
    stored_metrics = np.load(METRICS_PATH)
elif LEGACY_METRICS_PATH.exists():
    stored_metrics = np.load(LEGACY_METRICS_PATH)
    print("Loaded legacy numpy best metrics; they will be saved to the new directory on update.")

if stored_metrics is not None:
    best_accuracy = float(stored_metrics.get("best_accuracy", 0.0))
    print(f"Loaded previous global best accuracy: {best_accuracy*100:.2f}%")
else:
    best_accuracy = 0.0
run_best_accuracy = 0.0
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


    if test_accuracy > run_best_accuracy:
        run_best_accuracy = test_accuracy

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = {
            'W1': nn.W1.copy(),
            'b1': nn.b1.copy(),
            'W2': nn.W2.copy(),
            'b2': nn.b2.copy()
        }
        save_to_onnx(best_model, MODEL_PATH)
        np.savez(METRICS_PATH, best_accuracy=best_accuracy)
        print(f"New global best model saved! (Accuracy: {test_accuracy*100:.2f}%)")

print(f"Training completed! Run best accuracy: {run_best_accuracy*100:.2f}% | Global best accuracy: {best_accuracy*100:.2f}%")
