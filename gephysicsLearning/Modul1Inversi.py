import numpy as np
#import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv  # Importing inverse and pseudo-inverse

# Step 1: Data Observations (Depth `z` and Temperature `T`)
z = np.array([5, 16, 25, 40, 50, 60, 70, 80, 90, 100])  # Depth (meters)
T = np.array([35.4, 50.1, 77.3, 92.3, 137.6, 147.0, 180.8, 182.7, 188.5, 223.2])  # Temperature (Celsius)

print("Observed Data:")
for i in range(len(z)):
    print(f"Depth: {z[i]}m -> Temperature: {T[i]}°C")

# Step 2: Constructing the Kernel Matrix G and Vector d
G = np.zeros((len(z), 2))  # Creating a (10,2) matrix

for i in range(len(z)):
    G[i, 0] = 1   # First column for bias (intercept)
    G[i, 1] = z[i]  # Second column for depth values

d = T.reshape(len(T), 1)  # Reshape `T` into a column vector

print("\nKernel Matrix G:")
print(G)

print("\nVector d:")
print(d)

# Step 3: Compute Least-Squares Solution using (G^T G)^-1 G^T d
try:
    GTG_inv = inv(G.T @ G)  # Compute (G^T G)^-1
    print("\n(G^T G)^-1:")
    print(GTG_inv)
except np.linalg.LinAlgError:
    print("\n(G^T G) is singular, using pseudo-inverse instead.")
    GTG_inv = pinv(G.T @ G)  # Use pseudo-inverse if matrix is singular

m = GTG_inv @ G.T @ d  # Compute parameters (a0 and a1)
print(m)

# Extract parameters
a0, a1 = m.flatten()  # Convert to scalar values

print("\nComputed Model Parameters:")
print(f"Intercept (a0): {a0}")
print(f"Slope (a1): {a1}")

# Step 4: Compute Predictions Using Linear Model T = a0 + a1 * z
linear_predictions = a0 + a1 * z

print("\nPredicted Temperatures:")
for i in range(len(z)):
    print(f"Depth: {z[i]}m -> Predicted Temperature: {linear_predictions[i]:.2f}°C")

# Step 5: Plot the Observed Data vs. Model
# plt.scatter(z, T, color='red', label='Observed Data', marker='o')
# plt.plot(z, linear_predictions, color='blue', label='Least Squares Fit', linestyle='--')
# plt.grid(True)
# plt.legend()
# plt.xlabel('Depth (m)')
# plt.ylabel('Temperature (°C)')
# plt.title('Variation of Temperature with Depth')
# plt.show()
