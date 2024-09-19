import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from skimage.io import imread, imshow

def segment_image(image_path, k=3, n_init=1, max_iterations=40):
    image = imread(image_path)
    original_shape = image.shape

    pixels = image.reshape(-1, 3)
    pixels = pixels / 255.0
    
    # Correct the argument 'n_clusters' instead of 'k'
    kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iterations)
    labels = kmeans.fit_predict(pixels)
    
    # Segment the image based on cluster centroids
    segmented_image = np.zeros_like(pixels)
    for i, label in enumerate(labels):
        segmented_image[i] = kmeans.cluster_centers_[label]
    
    segmented_image = segmented_image.reshape(original_shape)
    segmented_image = (segmented_image * 255).astype(np.uint8)

    # Compute evaluation metrics
    silhouette_avg = silhouette_score(pixels, labels)
    db_index = davies_bouldin_score(pixels, labels)
    inertia = kmeans.inertia_

    # Plot the original and segmented images
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    imshow(image)

    # Segmented image
    plt.subplot(1, 2, 2)
    plt.title(f"Segmented Image with {k} colors")
    imshow(segmented_image)
    
    print(silhouette_avg)
    print(db_index)
    print(inertia)

    # Display the silhouette score, Davies-Bouldin index, and inertia on the segmented image
    plt.text(0, -20, f'Silhouette Score: {silhouette_avg:.4f}', fontsize=12, color='white', backgroundcolor='black', weight='bold')
    plt.text(0, -30, f'Davies-Bouldin Index: {db_index:.4f}', fontsize=12, color='white', backgroundcolor='black', weight='bold')
    plt.text(0, -40, f'Inertia: {inertia:.4f}', fontsize=12, color='white', backgroundcolor='black', weight='bold')

    plt.tight_layout()
    plt.show()

segment_image('.venv/rose.jpeg', k=3, n_init=1, max_iterations=40)

