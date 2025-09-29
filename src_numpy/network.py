import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_size):
        # initialize weights and biases
        self.W1 = np.random.randn(hidden_size, 784) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(10, hidden_size) * 0.1
        self.b2 = np.zeros((10, 1))

    def ReLU(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        shiftX = x - np.max(x, axis=0, keepdims=True)
        return np.exp(shiftX) / np.sum(np.exp(shiftX), axis=0, keepdims=True)

    def ReLU_derivative(self, x):
        return x > 0


    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (self.W2.T @ dZ2) * self.ReLU_derivative(Z1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_param(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.mean(predictions == Y.ravel())
    
    def get_loss(self, A2, Y):
        # Cross-entropy loss
        m = A2.shape[1]  # Number of samples
        log_likelihood = -np.log(A2[Y.ravel(), np.arange(m)] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss