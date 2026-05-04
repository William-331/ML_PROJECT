import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.1):

        # num_iterations: The maximum number of iterations for gradient descent
        # lambda_param: L2 regularization coefficient
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        
        # Weights and bias, initialized as None
        self.weights = None
        self.bias = None
        
        # Record the cost at each iteration for analysis
        self.cost_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)   # Use np.clip to limit the range of z to prevent numerical overflow errors when calculating the sigmoid function.
        return 1 / (1 + np.exp(-z))
    
    
    def predict_prob(self, X):
        # Calculate the linear combination of inputs and weights, then apply the sigmoid function to get probabilities.
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        # Predict class labels based on predicted probabilities and a specified threshold (default is 0.5).
        probs = self.predict_prob(X)
        # If the probability >= 0.5, predict as 1, otherwise 0
        return (probs >= threshold).astype(int)

    
    def fit(self, X, y):  # Train the model using Gradient Descent with L2 Regularization.
        #X: Feature matrix of shape (m, n) -> m samples, n features
        #y: True labels of shape (m,)
        m, n = X.shape       ## m: Number of samples, n: Number of features

        # Initialize parameters: Set weights to 0 array, bias to 0.
        self.weights = np.zeros(n)
        self.bias = 0

        # Start Gradient Descent iterations
        for i in range(self.num_iterations):

            # A. Forward Propagation
            # Calculate linear combination: z = X * weights + bias
            linear_model = np.dot(X, self.weights) + self.bias

            # Calculate predicted probabilities using Sigmoid
            h = self._sigmoid(linear_model)

            # B. Calculate Gradients
            # Calculate how much each parameter contributed to the error (h - y). 
            # The L2 term ensures no single weight dominates the model, preventing overfitting.
            dw = (1 / m) * np.dot(X.T, (h - y)) + (self.lambda_param / m) * self.weights      # Gradient of weights (dw)

            db = (1 / m) * np.sum(h - y)      # Gradient of bias (db)
            
            # C. Update Parameters
            # Adjust the parameters in the opposite direction of the gradient by learning rate.
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db    

            # D. Evaluation Metrics
            # Compute the overall penalized Loss (Cross-Entropy + L2 Penalty) to track convergence.
            epsilon = 1e-9  # Safety margin to prevent log(0) domain errors

            cross_entropy_cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
            l2_penalty = (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
            
            total_cost = cross_entropy_cost + l2_penalty
            self.cost_history.append(total_cost)
