"""
=============================================================================
LESSON 03: YOUR FIRST ML MODEL — LINEAR REGRESSION & GRADIENT DESCENT
=============================================================================
In this lesson, you will build Linear Regression completely from scratch
using only NumPy, understand the math behind Gradient Descent, and compare
your model directly against Scikit-Learn!

Run this file with:
    python 02_linear_regression_from_scratch/01_linear_regression.py
=============================================================================
"""

import numpy as np


def separator(title: str):
    print(f"\n{'=' * 22} {title} {'=' * 22}")


# ---------------------------------------------------------------------------
# 1. THE THEORY: PREDICTION, LOSS & GRADIENTS
# ---------------------------------------------------------------------------
separator("1. The Core Math of Linear Regression")
print("""
1. HYPOTHESIS (Forward Pass):
   y_pred = X @ W + b
   - X: Input features matrix of shape (N samples, D features)
   - W: Weights vector of shape (D, 1) -> how much each feature matters
   - b: Bias scalar -> baseline value when all features are 0

2. LOSS FUNCTION (Mean Squared Error - MSE):
   Loss = (1 / N) * sum((y_pred - y_true) ** 2)
   Measures how far our predictions are from reality on average.

3. OPTIMIZATION (Gradient Descent):
   We compute the slope (gradient) of the loss with respect to W and b:
   - dW = (2 / N) * X.T @ (y_pred - y)
   - db = (2 / N) * sum(y_pred - y)
   
   Then update our parameters in the opposite direction of the slope:
   - W = W - (learning_rate * dW)
   - b = b - (learning_rate * db)
""")


# ---------------------------------------------------------------------------
# 2. BUILDING THE SCRATCH MODEL CLASS
# ---------------------------------------------------------------------------
separator("2. Linear Regression Class From Scratch")


class LinearRegressionScratch:
    """
    Linear Regression model trained using Batch Gradient Descent.
    Built entirely with NumPy vectorization!
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on feature matrix X and target labels y.
        """
        N, D = X.shape

        # Ensure y is a 2D column vector (N, 1) for proper matrix broadcasting
        y = y.reshape(-1, 1)

        # 1. Initialize weights with zeros (or small random numbers)
        self.weights = np.zeros((D, 1))
        self.bias = 0.0
        self.loss_history = []

        # 2. Gradient Descent Loop
        for epoch in range(1, self.epochs + 1):
            # A. Forward pass: compute predictions
            y_pred = X @ self.weights + self.bias  # Shape: (N, 1)

            # B. Compute current Mean Squared Error (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # C. Compute gradients (derivatives of MSE)
            error = y_pred - y                     # Shape: (N, 1)
            dW = (2 / N) * (X.T @ error)           # Shape: (D, 1)
            db = (2 / N) * np.sum(error)           # Scalar

            # D. Update weights and bias (take a step downhill)
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            # Log progress every 200 epochs
            if epoch % 200 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{self.epochs} | Loss (MSE): {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new feature data X.
        """
        if self.weights is None:
            raise ValueError("Model is not trained yet. Call .fit() first!")
        return X @ self.weights + self.bias

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R^2 (Coefficient of Determination).
        1.0 means perfect predictions, 0.0 means predicting the mean.
        Formula: 1 - (SS_residual / SS_total)
        """
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - (ss_res / ss_tot))


# ---------------------------------------------------------------------------
# 3. GENERATING SYNTHETIC DATA & TESTING
# ---------------------------------------------------------------------------
separator("3. Training On A Dataset")

# Let's create a synthetic housing dataset:
# True formula: Price = 3.5 * HouseSize + 2.0 * Bedrooms + 15.0 (noise added)
np.random.seed(42)
N_SAMPLES = 200

# 2 features: Normalized house size and number of bedrooms
X = np.random.randn(N_SAMPLES, 2)
true_weights = np.array([[3.5], [2.0]])
true_bias = 15.0

# y = X @ W + b + random Gaussian noise
noise = np.random.randn(N_SAMPLES, 1) * 0.5
y = (X @ true_weights + true_bias + noise).flatten()

# Split into 80% Train, 20% Test
split_idx = int(N_SAMPLES * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Dataset generated: {N_SAMPLES} samples, 2 features.")
print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

# Train our custom scratch model
print("--- Training Scratch Linear Regression ---")
model_scratch = LinearRegressionScratch(learning_rate=0.05, epochs=1000)
model_scratch.fit(X_train, y_train)

# Evaluate on unseen test data
r2_scratch = model_scratch.r2_score(X_test, y_test)
print(f"\nScratch Model Test R^2 Score: {r2_scratch:.4f}")
print(f"Learned Weights:\n{model_scratch.weights.flatten()}")
print(f"Learned Bias: {model_scratch.bias:.4f}")


# ---------------------------------------------------------------------------
# 4. BENCHMARK AGAINST PRODUCTION SCIKIT-LEARN
# ---------------------------------------------------------------------------
separator("4. Comparison Against Scikit-Learn")

try:
    from sklearn.linear_model import LinearRegression

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)

    sklearn_r2 = sklearn_model.score(X_test, y_test)
    print("--- Scikit-Learn Model Results ---")
    print(f"Scikit-Learn Test R^2 Score: {sklearn_r2:.4f}")
    print(f"Scikit-Learn Weights:        {sklearn_model.coef_}")
    print(f"Scikit-Learn Bias:           {sklearn_model.intercept_:.4f}")

    print("\n--- Accuracy Comparison ---")
    diff_w = np.max(np.abs(model_scratch.weights.flatten() - sklearn_model.coef_))
    diff_b = abs(model_scratch.bias - sklearn_model.intercept_)
    print(f"Max difference in weights: {diff_w:.6f}")
    print(f"Difference in bias:        {diff_b:.6f}")
    if diff_w < 0.01 and diff_b < 0.01:
        print("--> SUCCESS: Your scratch implementation matches Scikit-Learn's output!")
except ImportError:
    print("Note: scikit-learn is not installed in this environment yet.")


# ---------------------------------------------------------------------------
# 5. MINI EXERCISE FOR YOU
# ---------------------------------------------------------------------------
separator("5. Mini Experiment Challenge")
print("""
CHALLENGE:
1. In the terminal, run this script and observe how the loss decreases.
2. In the code above, try changing the learning_rate in:
     model_scratch = LinearRegressionScratch(learning_rate=0.05, epochs=1000)
   - What happens if learning_rate = 1.5? (Too high -> gradients explode / loss goes to infinity)
   - What happens if learning_rate = 0.0001? (Too low -> learns too slowly)
   Finding the right learning rate is one of the most critical skills in ML!
""")

if __name__ == "__main__":
    pass
