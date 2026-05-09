import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.1,
                 penalty='l2'):

        # num_iterations: The maximum number of iterations for gradient descent
        # lambda_param: Regularization coefficient (controls strength of penalty)
        # penalty: 'l1' (Lasso, sparse weights) or 'l2' (Ridge, small weights)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.penalty = penalty

        # Weights and bias, initialized as None
        self.weights = None
        self.bias = None

        # Record the cost at each iteration for analysis
        self.cost_history = []

    def _soft_threshold(self, w, threshold):
        # Proximal operator for L1: shrink each weight towards zero by threshold.
        # w = sign(w) * max(0, |w| - threshold)
        return np.sign(w) * np.maximum(0, np.abs(w) - threshold)

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

    
    def fit(self, X, y):  # Train the model using Gradient Descent with L1 or L2 Regularization.
        #X: Feature matrix of shape (m, n) -> m samples, n features
        #y: True labels of shape (m,)
        m, n = X.shape       ## m: Number of samples, n: Number of features

        # Initialize parameters: Set weights to 0 array, bias to 0.
        self.weights = np.zeros(n)
        self.bias = 0

        # Start Gradient Descent iterations
        for i in range(self.num_iterations):

            # A. Forward Propagation
            linear_model = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(linear_model)

            # B. Compute Cross-Entropy Gradient
            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)

            # L2 penalty: gradient = (lambda / m) * w (shrinks all weights towards zero)
            if self.penalty == 'l2':
                dw += (self.lambda_param / m) * self.weights

            # C. Parameter Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # L1 penalty: proximal gradient step (soft thresholding).
            # Unlike L2, this pushes unimportant weights exactly to zero -> automatic feature selection.
            if self.penalty == 'l1':
                threshold = self.learning_rate * self.lambda_param
                self.weights = self._soft_threshold(self.weights, threshold)

            # D. Compute and Record Cost
            epsilon = 1e-9  # Safety margin to prevent log(0) domain errors
            cross_entropy_cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

            if self.penalty == 'l2':
                reg_cost = (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
            elif self.penalty == 'l1':
                reg_cost = (self.lambda_param / m) * np.sum(np.abs(self.weights))
            else:
                reg_cost = 0

            total_cost = cross_entropy_cost + reg_cost
            self.cost_history.append(total_cost)

    def get_important_features(self, threshold=1e-5):
        # Return indices and weights of features with non-zero weight (useful after L1 training).
        # Returns (indices, weights) sorted by |weight| descending.
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        mask = np.abs(self.weights) > threshold
        indices = np.where(mask)[0]
        weights = self.weights[mask]
        # Sort by absolute importance (largest -> smallest)
        sort_order = np.argsort(np.abs(weights))[::-1]
        return indices[sort_order], weights[sort_order]
