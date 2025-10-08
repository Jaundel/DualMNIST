import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MNISTCNN

# Load data (robust to current working directory)
print("Loading data...")
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "pytorch"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "mnist_pytorch.onnx"
METRICS_PATH = MODELS_DIR / "best_metrics.npz"
CSV_DIR = ROOT / "data" / "MNIST_CSV"

train_df = pd.read_csv(CSV_DIR / "mnist_train.csv", header=None)
test_df = pd.read_csv(CSV_DIR / "mnist_test.csv", header=None)

# Extract and normalize data
y_train = train_df.iloc[:, 0].values.astype(np.int64)
X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
y_test = test_df.iloc[:, 0].values.astype(np.int64)
X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0

IMAGE_SHAPE = (1, 28, 28)

X_train_tensor = torch.from_numpy(X_train.reshape(-1, *IMAGE_SHAPE))
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test.reshape(-1, *IMAGE_SHAPE))
y_test_tensor = torch.from_numpy(y_test)

# Initialize model and training utilities
learning_rate = 0.1
epochs = 10
batch_size = 64
input_shape = X_train_tensor.shape[1:]


def build_model():
    return MNISTCNN()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)


def save_to_onnx(state_dict, path, input_shape):
    export_model = build_model()
    export_model.load_state_dict(state_dict)
    export_model.eval()
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(path),
        input_names=["X"],
        output_names=["y"],
        dynamic_axes={"X": {0: "batch"}, "y": {0: "batch"}},
    )


# Track best model (persisted globally)
if METRICS_PATH.exists():
    stored_metrics = np.load(METRICS_PATH)
    best_accuracy = float(stored_metrics.get("best_accuracy", 0.0))
    print(f"Loaded previous global best accuracy: {best_accuracy*100:.2f}%")
else:
    best_accuracy = 0.0
run_best_accuracy = 0.0
best_state = None

# Training loop
num_batches = len(train_loader)
for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()

    epoch_train_loss = 0.0
    epoch_train_acc = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_accuracy = (preds == targets).float().mean().item()
        batch_loss = loss.item()

        epoch_train_loss += batch_loss
        epoch_train_acc += batch_accuracy

        if batch_idx % 200 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{num_batches} - "
                f"Loss: {batch_loss:.4f} - Acc: {batch_accuracy*100:.2f}%"
            )

    avg_train_loss = epoch_train_loss / num_batches
    avg_train_acc = epoch_train_acc / num_batches

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()
        test_predictions = test_outputs.argmax(dim=1).cpu()
        test_accuracy = (test_predictions == y_test_tensor).float().mean().item()

    epoch_time = time.time() - epoch_start_time

    print(
        f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - "
        f"Train Acc: {avg_train_acc*100:.2f}% - Test Loss: {test_loss:.4f} - "
        f"Test Acc: {test_accuracy*100:.2f}% - Time: {epoch_time:.1f}s"
    )

    if test_accuracy > run_best_accuracy:
        run_best_accuracy = test_accuracy

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_to_onnx(best_state, MODEL_PATH, input_shape)
        np.savez(METRICS_PATH, best_accuracy=best_accuracy)
        print(f"New global best model saved! (Accuracy: {test_accuracy*100:.2f}%)")

if best_state is not None:
    model.load_state_dict(best_state)

print(
    f"Training completed! Run best accuracy: {run_best_accuracy*100:.2f}% | "
    f"Global best accuracy: {best_accuracy*100:.2f}%"
)
