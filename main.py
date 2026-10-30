import numpy as np
from Neuron import Neuron

# Features: [Hours Studied, Hours Slept] (Scaled from 0 to 1)
X = np.array([
    [0.1, 0.2],  # Low study, low sleep -> Fail (0)
    [0.2, 0.8],  # Low study, high sleep -> Fail (0)
    [0.1, 0.9],  # Low study, high sleep -> Fail (0)
    [0.3, 0.3],  # Low study, low sleep -> Fail (0)
    [0.4, 0.5],  # Medium study, medium sleep -> Fail (0)
    [0.6, 0.2],  # High study, low sleep -> Pass (1)
    [0.7, 0.8],  # High study, high sleep -> Pass (1)
    [0.8, 0.6],  # High study, medium sleep -> Pass (1)
    [0.9, 0.9],  # High study, high sleep -> Pass (1)
    [0.8, 0.3]   # High study, low sleep -> Pass (1)
], dtype=float)

# Targets: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)

# Initialize your neuron with 2 weights and a bias
neuron = Neuron(weights=np.array([0.1, -0.1]), bias=0.0)

print("--- Starting Training ---")
neuron.train_za_neron_boiii(X, y, learning_rate=0.2, epochs=2000)

print("\n--- Testing Predictions After Training ---")
for inputs in X:
    neuron.inputs = inputs
    prediction = neuron.forward()
    print(f"Inputs: {inputs} | Prediction: {prediction:.4f} | Rounded: {round(prediction)}")