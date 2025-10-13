import numpy as np


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)   # assumes x is already sigmoid(x)


# Neural Network Class

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        self.lr = lr

    def forward(self, X):
        # Input -> Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Hidden -> Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear output (regression task)
        
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]  # batch size

        # Error at output
        error = output - y  # dL/dy_pred
        
        # Gradients for W2, b2
        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m

        # Backprop into hidden layer
        d_hidden = np.dot(error, self.W2.T) * sigmoid_derivative(self.a1)

        # Gradients for W1, b1
        dW1 = np.dot(X.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)


# Training Data (Addition)

X = np.array([
    [2, 2],
    [3, 4],
    [1, 5],
    [0, 6],
    [7, 3],
    [10, 15],
    [4, 8]
], dtype=float)

y = np.array([[4], [7], [6], [6], [10], [25], [12]], dtype=float)


# Train the Model

nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=1, lr=0.01)
nn.train(X, y, epochs=10000)


# Test Predictions

test_data = np.array([[5, 9], [12, 7], [6, 6]], dtype=float)
predictions = nn.predict(test_data)

print("\nTest Predictions:")
for i, p in enumerate(predictions):
    print(f"{test_data[i][0]} + {test_data[i][1]} ≈ {p[0]:.2f}")
