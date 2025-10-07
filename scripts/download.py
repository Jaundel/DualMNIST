
from pathlib import Path
import numpy as np
from torchvision import datasets

OUT = Path("../data/MNIST_CSV")
OUT.mkdir(parents=True, exist_ok=True)

def save_csv(ds, path):

    X = ds.data.numpy().reshape(len(ds), -1)
    y = ds.targets.numpy().reshape(-1, 1)
    np.savetxt(path, np.hstack([y, X]), fmt="%d", delimiter=",")
    print(f"Wrote {path}  with shape {y.shape[0]}x785")

for name, is_train in [("mnist_train.csv", True), ("mnist_test.csv", False)]:
    fp = OUT / name
    if not fp.exists():
        ds = datasets.MNIST(root="data/raw_mnist", train=is_train, download=True)
        save_csv(ds, fp)
    else:
        print(f"Found {name}, skipping.")

print("CSV dataset is ready.")
