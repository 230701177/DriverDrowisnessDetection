import numpy as np
from sklearn import svm
import joblib
import time

print("Loading data...")

# Load features and labels
X = np.load("features.npy")
y = np.load("labels.npy")

print(f"Dataset loaded: {X.shape[0]} samples")

# Initialize model
model = svm.SVC(kernel='linear', verbose=True)

print("Training started...\n")

start_time = time.time()

# Train model
model.fit(X, y)

end_time = time.time()

print("\nTraining completed!")

# Save model
joblib.dump(model, "model.pkl")

print(f"Model saved as 'model.pkl'")
print(f"Training time: {end_time - start_time:.2f} seconds")
print("Model Trained Successfully ✅")