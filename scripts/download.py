import urllib.request
from pathlib import Path

# --- Configuration ---
# Publicly hosted CSV versions of MNIST
URLS = {
    "mnist_train.csv": "https://pjreddie.com/media/files/mnist_train.csv",
    "mnist_test.csv": "https://pjreddie.com/media/files/mnist_test.csv"
}
DATA_PATH = Path("data/MNIST_CSV")

# --- Script ---
DATA_PATH.mkdir(parents=True, exist_ok=True)

for filename, url in URLS.items():
    filepath = DATA_PATH / filename
    
    if not filepath.exists():
        print(f"Downloading {filename}... (This may take a moment)")
        urllib.request.urlretrieve(url, filepath)

print("CSV dataset is ready.")