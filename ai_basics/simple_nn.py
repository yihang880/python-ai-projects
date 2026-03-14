import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def forward(self, X):
        # Hidden layer
        self.hidden_layer_input = np.dot(X, self.W1) + self.b1
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.W2) + self.b2
        predictions = self.sigmoid(self.output_layer_input)
        return predictions
        
    def backward(self, X, y, predictions, learning_rate):
        # Output layer error
        error_output = y - predictions
        delta_output = error_output * self.sigmoid_derivative(predictions)
        
        # Hidden layer error
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.W2 += np.dot(self.hidden_layer_output.T, delta_output) * learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.W1 += np.dot(X.T, delta_hidden) * learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y, predictions, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - predictions))
                print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    # Example usage:
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]]) # XOR problem
    
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    print("
Final Predictions:")
    print(nn.forward(X))
