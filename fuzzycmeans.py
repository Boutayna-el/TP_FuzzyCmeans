import numpy as np
import matplotlib.pyplot as plt

# --- FCM Functions (identical to before) ---

def initialize_membership(n_samples, n_clusters):
    """
    Step 1: Initialize the membership matrix U.
    Based on[cite: 14].
    Each row must sum to 1.
    """
    matrix = np.random.rand(n_samples, n_clusters)
    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
    return matrix

def calculate_centers(data, U, m, n_clusters):
    """
    Step 2: Calculate the cluster centers C.
    Implements formula from[cite: 15, 16].
    """
    centers = []
    for j in range(n_clusters):
        # (u_ij^m)
        u_m = U[:, j] ** m
        # Numerator: sum(u_ij^m * x_i)
        numerator = np.dot(u_m, data)
        # Denominator: sum(u_ij^m)
        denominator = np.sum(u_m)
        centers.append(numerator / denominator)
    return np.array(centers)

def update_membership(data, centers, m):
    """
    Step 3: Update the membership matrix U.
    Implements formula from[cite: 17, 18].
    """
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    U_new = np.zeros((n_samples, n_clusters))
    
    # Calculate distances
    distances = np.zeros((n_samples, n_clusters))
    for j in range(n_clusters):
        # Using simple absolute difference for 1D data
        distances[:, j] = np.abs(data - centers[j])
    
    # Handle potential division by zero (if a data point is on a center)
    distances[distances == 0] = 1e-9 
    
    power_val = 2.0 / (m - 1.0) # [cite: 18]
    
    for i in range(n_samples):
        for j in range(n_clusters):
            sum_val = 0.0
            for k in range(n_clusters):
                # This is the implementation of the formula [cite: 18]
                sum_val += (distances[i, j] / distances[i, k]) ** power_val
            U_new[i, j] = 1.0 / sum_val
            
    return U_new

def fuzzy_c_means(data, n_clusters, m, epsilon):
    """
    Main FCM algorithm loop.
    Iterates steps 2 and 3 until convergence[cite: 19].
    """
    # 1. Initialize U [cite: 14]
    n_samples = data.shape[0]
    U = initialize_membership(n_samples, n_clusters)
    
    iteration = 0
    while True:
        iteration += 1
        U_old = U.copy()
        
        # 2. Calculate centers C [cite: 15]
        centers = calculate_centers(data, U, m, n_clusters)
        
        # 3. Update U [cite: 17]
        U = update_membership(data, centers, m)
        
        # 4. Check for convergence [cite: 19]
        max_change = np.max(np.abs(U - U_old))
        if max_change < epsilon:
            break
            
    return U, centers, iteration

# --- Main execution ---

# Parameters
n_clusters = 2  # As suggested [cite: 119]
m = 2.0         # Fuzziness coefficient (m > 1) [cite: 7]
epsilon = 0.01  # Termination criterion [cite: 11]
image_path = 'milky-way-nvg.jpg'

# 1. Load and prepare the image
try:
    img = plt.imread(image_path)
    if img.ndim == 3: # Convert to grayscale if it's color
        img = img.mean(axis=2)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

original_shape = img.shape
# Flatten the 2D image into a 1D array of pixel intensities
data = img.flatten()

print(f"Applying FCM to '{image_path}'...")
print(f"Parameters: K={n_clusters}, m={m}, epsilon={epsilon}")

# 2. Run the FCM algorithm
U, centers, iterations = fuzzy_c_means(data, n_clusters, m, epsilon)

print(f"Convergence reached in {iterations} iterations.")
# Sort centers to consistently identify 'dark' and 'bright'
sorted_indices = np.argsort(centers)
sorted_centers = centers[sorted_indices]
print(f"Final cluster centers (pixel intensity): {sorted_centers}")

# Re-order U columns to match sorted centers
U_sorted = U[:, sorted_indices]

# 3. Create Heatmap visualization 
# We'll map the membership to the *brightest* cluster (the second cluster after sorting)
bright_cluster_index = 1 
membership_heatmap = U_sorted[:, bright_cluster_index].reshape(original_shape)

# 4. Create Hard Segmentation (defuzzified)
# This is what I showed before
labels = np.argmax(U_sorted, axis=1)
segmented_image = labels.reshape(original_shape)

# 5. Visualize all three results
plt.figure(figsize=(18, 6))

# Plot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Plot 2: Hard Segmentation
plt.subplot(1, 3, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title(f'Hard Segmentation (K={n_clusters})')
plt.axis('off')

# Plot 3: Fuzzy Heatmap
plt.subplot(1, 3, 3)
# Use a sequential colormap like 'viridis' or 'hot' for the heatmap
im = plt.imshow(membership_heatmap, cmap='hot')
plt.title('Fuzzy Heatmap (Membership to Bright Cluster)')
plt.axis('off')
plt.colorbar(im, fraction=0.046, pad=0.04) # Add a color bar to show membership values

plt.tight_layout()
plt.savefig('fcm_segmentation_with_heatmap.png')
print("Saved visualization to 'fcm_segmentation_with_heatmap.png'")