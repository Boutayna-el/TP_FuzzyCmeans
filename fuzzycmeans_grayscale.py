import numpy as np
import matplotlib.pyplot as plt
import os

#Question 1: Fuzzy C-Means Clustering Implementation

def initialize_membership(n_samples, n_clusters):
    """
    Step 1: Initialize the membership matrix U.
    """
    matrix = np.random.rand(n_samples, n_clusters)
    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
    return matrix

def calculate_centers(data, U, m, n_clusters):
    """
    Step 2: Calculate the cluster centers C.
    """
    n_features = data.shape[1]
    centers = np.zeros((n_clusters, n_features))
    
    for j in range(n_clusters):
        u_m = U[:, j] ** m
        u_m_expanded = u_m[:, np.newaxis] 
     
        numerator = np.sum(u_m_expanded * data, axis=0)
        denominator = np.sum(u_m)
        
        centers[j] = numerator / denominator
    return centers

def update_membership(data, centers, m):
    """
    Step 3: Update the membership matrix U.
    """
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    
    distances = np.zeros((n_samples, n_clusters))
    
    for j in range(n_clusters):
        distances[:, j] = np.linalg.norm(data - centers[j], axis=1)
    
    distances[distances == 0] = 1e-9 
    
    power_val = 2.0 / (m - 1.0)
    
    inv_dist_powered = (1.0 / distances) ** power_val
    sum_inv_dist = np.sum(inv_dist_powered, axis=1, keepdims=True)
    
    U_new = inv_dist_powered / sum_inv_dist
            
    return U_new

def fuzzy_c_means(data, n_clusters, m, epsilon):
    """
    Main FCM algorithm loop.
    """
    n_samples = data.shape[0]
    U = initialize_membership(n_samples, n_clusters)
    
    iteration = 0
    while True:
        iteration += 1
        U_old = U.copy()
        centers = calculate_centers(data, U, m, n_clusters)
        U = update_membership(data, centers, m)
        max_change = np.max(np.abs(U - U_old))
        if max_change < epsilon:
            break
            
    return U, centers, iteration


#Question 2: Applying FCM to Grayscale Image Segmentation 
# --- Parameters ---
m = 2.0        
epsilon = 0.01   
image_path = 'milky-way-nvg.jpg' 
cluster_list = [2]               
cluster_heatmap_to_show = 0    

try:
    img = plt.imread(image_path)
    
    if img.ndim == 3:
        img = img.mean(axis=2) 
        
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

original_shape = img.shape
data = img.reshape(-1, 1) 
print(f"Image loaded: {original_shape}, reshaped to {data.shape} for clustering.")

for n_clusters in cluster_list:
    
    print(f"\n--- Applying FCM for K={n_clusters} ---")
    print(f"Parameters: m={m}, epsilon={epsilon}")

    U, centers, iterations = fuzzy_c_means(data, n_clusters, m, epsilon)

    print(f"Convergence reached in {iterations} iterations.")
    print(f"Final cluster centers (pixel intensity): \n{centers.flatten()}")

    labels = np.argmax(U, axis=1)
    
    segmented_data = centers[labels]
    segmented_image = segmented_data.reshape(original_shape) 

    sorted_indices = np.argsort(centers.flatten())
    heatmap_cluster_index = sorted_indices[0] 
    
    membership_heatmap = U[:, heatmap_cluster_index].reshape(original_shape)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image, cmap='gray') 
    plt.title(f'Hard Segmentation (K={n_clusters})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    im = plt.imshow(membership_heatmap, cmap='hot')
    plt.title(f'Fuzzy Heatmap (Membership to Dark Cluster)')
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04) 

    plt.tight_layout()
    
    save_filename = f'fcm_grayscale_segmentation_K{n_clusters}.png'
    plt.savefig(save_filename)
    print(f"Saved visualization to '{save_filename}'")

print("\nAll segmentations complete.")