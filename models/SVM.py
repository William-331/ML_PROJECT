import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.1,
                 kernel='linear', gamma=0.1, class_weight=None, optimizer='sgd'):

        # num_iterations: The maximum number of iterations for gradient descent
        # lambda_param: L2 regularization coefficient
        # kernel: 'linear' or 'rbf_exact'
        # gamma: Controls the width of the RBF kernel (small gamma = smoother boundary)
        # class_weight: 'balanced' to auto-weight by inverse class frequency, or None
        # optimizer: 'sgd' or 'adam'
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.optimizer = optimizer

        # Model parameters
        self.weights = None    # primal weights (linear kernel)
        self.alpha = None      # Representer Theorem coefficients (rbf_exact kernel)
        self.bias = None

        # For rbf_exact prediction
        self.X_train = None
        self.support_idx = None

        # Record the cost at each iteration for analysis
        self.cost_history = []

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    def _rbf_kernel_matrix(self, X1, X2):
        # K(x, z) = exp(-gamma * ||x - z||^2)
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def decision_function(self, X):
        if self.kernel == 'rbf_exact':
            # f(x) = sum_i alpha_i * K(x_i, x) + b
            K = self._rbf_kernel_matrix(self.X_train[self.support_idx], X)
            return np.dot(self.alpha[self.support_idx].T, K) + self.bias
        else:
            return np.dot(X, self.weights) / X.shape[1] + self.bias

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y):
        #X: Feature matrix of shape (m, n) -> m samples, n features
        #y: True labels of shape (m,) with values 0 or 1
        m, n = X.shape

        # Convert labels from {0,1} to {-1,+1} for hinge loss
        y_svm = np.where(y == 0, -1, 1)

        # Class-balanced sample weights
        if self.class_weight == 'balanced':
            n_pos = np.sum(y_svm == 1)
            n_neg = np.sum(y_svm == -1)
            w_pos = m / (2 * max(n_pos, 1))
            w_neg = m / (2 * max(n_neg, 1))
            sample_weights = np.where(y_svm == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(m)

        # ---- RBF Kernel: Representer Theorem ----
        if self.kernel == 'rbf_exact':
            self.X_train = X.copy()
            K = self._rbf_kernel_matrix(X, X)   # shape (m, m)

            # Initialize alpha
            self.alpha = np.zeros(m)
            self.bias = 0

            # Adam state
            if self.optimizer == 'adam':
                m_a, v_a = np.zeros(m), np.zeros(m)
                m_b_adam, v_b_adam = 0.0, 0.0
                beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

            for t in range(1, self.num_iterations + 1):
                # A. Forward pass
                f = np.dot(K, self.alpha) + self.bias

                # B. Sub-Gradient
                margin = 1 - y_svm * f
                violation_mask = margin > 0
                weighted_mask = y_svm * violation_mask * sample_weights

                dalpha = -(1 / m) * np.dot(K, weighted_mask) + (self.lambda_param / m) * np.dot(K, self.alpha)
                dbias = -(1 / m) * np.sum(weighted_mask)

                # C. Update
                if self.optimizer == 'adam':
                    m_a = beta1 * m_a + (1 - beta1) * dalpha
                    v_a = beta2 * v_a + (1 - beta2) * dalpha ** 2
                    m_b_adam = beta1 * m_b_adam + (1 - beta1) * dbias
                    v_b_adam = beta2 * v_b_adam + (1 - beta2) * dbias ** 2
                    m_a_hat = m_a / (1 - beta1 ** t)
                    v_a_hat = v_a / (1 - beta2 ** t)
                    m_b_hat = m_b_adam / (1 - beta1 ** t)
                    v_b_hat = v_b_adam / (1 - beta2 ** t)
                    self.alpha -= self.learning_rate * m_a_hat / (np.sqrt(v_a_hat) + eps_adam)
                    self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps_adam)
                else:
                    self.alpha -= self.learning_rate * dalpha
                    self.bias -= self.learning_rate * dbias

                # D. Cost
                hinge_loss = np.sum(np.maximum(0, margin) * sample_weights) / m
                l2_penalty = (self.lambda_param / (2 * m)) * (self.alpha @ K @ self.alpha)
                self.cost_history.append(hinge_loss + l2_penalty)

            self.support_idx = np.where(np.abs(self.alpha) > 1e-5)[0]
            if len(self.support_idx) == 0:
                self.support_idx = np.arange(m)

        # ---- Linear Kernel: Primal Optimization ----
        else:
            X_fit = X / np.sqrt(n)  # Scale to keep gradient stable
            self._fit_primal(X_fit, y_svm, sample_weights)

    def _fit_primal(self, X_fit, y_svm, sample_weights):
        # Primal sub-gradient descent for linear kernel
        m, n_params = X_fit.shape

        self.weights = np.zeros(n_params)
        self.bias = 0

        if self.optimizer == 'adam':
            m_w, v_w = np.zeros(n_params), np.zeros(n_params)
            m_b, v_b = 0.0, 0.0
            beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.num_iterations + 1):
            # A. Forward pass
            f = np.dot(X_fit, self.weights) + self.bias

            # B. Sub-Gradient
            margin = 1 - y_svm * f
            violation_mask = margin > 0
            weighted_mask = y_svm * violation_mask * sample_weights

            dw = -(1 / m) * np.dot(X_fit.T, weighted_mask) + (self.lambda_param / m) * self.weights
            db = -(1 / m) * np.sum(weighted_mask)

            # C. Update
            if self.optimizer == 'adam':
                m_w = beta1 * m_w + (1 - beta1) * dw
                v_w = beta2 * v_w + (1 - beta2) * dw ** 2
                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * db ** 2
                m_w_hat = m_w / (1 - beta1 ** t)
                v_w_hat = v_w / (1 - beta2 ** t)
                m_b_hat = m_b / (1 - beta1 ** t)
                v_b_hat = v_b / (1 - beta2 ** t)
                self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
            else:
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # D. Cost
            hinge_loss = np.sum(np.maximum(0, margin) * sample_weights) / m
            l2_penalty = (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
            self.cost_history.append(hinge_loss + l2_penalty)

    def get_support_vector_count(self):
        if self.support_idx is not None:
            return len(self.support_idx)
        return 0
