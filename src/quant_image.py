# src/quant_image.py
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import CSV_FILE, IMG_DIR, IMG_SIZE, QUANT_LEVELS, PAD_TO
from utils import normalize_and_quantize, row_to_image, save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


def correlation_based_mapping(X, img_size):
    """Map features to 2D grid based on correlation structure"""
    print("Computing correlation-based feature mapping...")
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Use spectral embedding to map features to 2D space
    from sklearn.manifold import SpectralEmbedding
    
    # Convert correlation to distance (higher correlation = closer)
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Use spectral embedding to find 2D coordinates for features
    n_features = X.shape[1]
    if n_features < 3:
        # For very few features, use simple arrangement
        coords = [(i, 0) for i in range(n_features)]
    else:
        embedding = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=42)
        try:
            feature_coords_2d = embedding.fit_transform(1 - distance_matrix)
            
            # Scale coordinates to image grid
            x_coords = ((feature_coords_2d[:, 0] - feature_coords_2d[:, 0].min()) / 
                       (feature_coords_2d[:, 0].max() - feature_coords_2d[:, 0].min() + 1e-8))
            y_coords = ((feature_coords_2d[:, 1] - feature_coords_2d[:, 1].min()) / 
                       (feature_coords_2d[:, 1].max() - feature_coords_2d[:, 1].min() + 1e-8))
            
            x_coords = (x_coords * (img_size - 1)).astype(int)
            y_coords = (y_coords * (img_size - 1)).astype(int)
            
            coords = list(zip(x_coords, y_coords))
        except Exception as e:
            print(f"Spectral embedding failed: {e}. Using simple grid arrangement.")
            # Fallback to simple grid arrangement
            coords = []
            for i in range(n_features):
                x = i % img_size
                y = i // img_size
                coords.append((x, y))
    
    return coords


def tsne_grid_encoding(X, img_size, perplexity=30):
    """Use t-SNE to arrange features in 2D grid based on similarity"""
    print("Computing t-SNE-based feature arrangement...")
    
    n_features = X.shape[1]
    
    if n_features < 4:
        # For very few features, use simple arrangement
        coords = [(i, 0) for i in range(n_features)]
        return coords
    
    # Use t-SNE on feature vectors (transposed)
    perplexity = min(perplexity, (n_features - 1) // 3)
    
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                   init='pca', learning_rate='auto')
        feature_coords_2d = tsne.fit_transform(X.T)  # Features as samples
        
        # Scale coordinates to image grid
        x_coords = ((feature_coords_2d[:, 0] - feature_coords_2d[:, 0].min()) / 
                   (feature_coords_2d[:, 0].max() - feature_coords_2d[:, 0].min() + 1e-8))
        y_coords = ((feature_coords_2d[:, 1] - feature_coords_2d[:, 1].min()) / 
                   (feature_coords_2d[:, 1].max() - feature_coords_2d[:, 1].min() + 1e-8))
        
        x_coords = (x_coords * (img_size - 1)).astype(int)
        y_coords = (y_coords * (img_size - 1)).astype(int)
        
        coords = list(zip(x_coords, y_coords))
        
    except Exception as e:
        print(f"t-SNE failed: {e}. Using correlation-based mapping.")
        coords = correlation_based_mapping(X, img_size)
    
    return coords


def create_structured_image(row, coords, img_size):
    """Create image using structured feature placement"""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Handle case where we have more features than coordinates
    n_features = len(row)
    n_coords = len(coords)
    
    for i, value in enumerate(row):
        if i < n_coords:
            x, y = coords[i]
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_size - 1))
            y = max(0, min(y, img_size - 1))
            img[y, x] = value  # Note: numpy arrays are [row, col] = [y, x]
        else:
            # For extra features, place them in remaining spots
            remaining_spot = (i % img_size, (i // img_size) % img_size)
            img[remaining_spot[1], remaining_spot[0]] = value
    
    return img


def gramian_angular_field(X, method='summation'):
    """Convert time series to Gramian Angular Field (adapted for tabular data)"""
    print(f"Computing Gramian Angular Field ({method})...")
    
    images = []
    for row in tqdm(X, desc="Converting to GAF"):
        # Normalize to [-1, 1] range
        row_norm = 2 * (row - row.min()) / (row.max() - row.min() + 1e-8) - 1
        
        # Compute angles
        angles = np.arccos(row_norm)
        
        # Create Gramian matrix
        n = len(row)
        if method == 'summation':
            gaf = np.cos(angles[:, np.newaxis] + angles[np.newaxis, :])
        elif method == 'difference':
            gaf = np.sin(angles[:, np.newaxis] - angles[np.newaxis, :])
        else:
            raise ValueError("Method must be 'summation' or 'difference'")
        
        # Normalize to [0, 255] for image
        gaf_normalized = ((gaf + 1) / 2 * 255).astype(np.uint8)
        images.append(gaf_normalized)
    
    return images


def build_and_save_images(csv_path=CSV_FILE, out_dir=IMG_DIR, img_size=IMG_SIZE, 
                         n_levels=QUANT_LEVELS, method='simple'):
    """
    Build and save images with different encoding methods
    
    Args:
        method: 'simple', 'correlation', 'tsne', 'gaf_sum', 'gaf_diff'
    """
    df = pd.read_csv(csv_path, sep=";")

    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    y = y.astype(str).str.strip()
    y_bin = y.apply(lambda v: 1 if v.lower() == "dropout" else 0).values

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    print(f"Using encoding method: {method}")
    
    if method in ['gaf_sum', 'gaf_diff']:
        # Gramian Angular Field methods don't use quantization
        gaf_method = 'summation' if method == 'gaf_sum' else 'difference'
        images = gramian_angular_field(X, method=gaf_method)
        
        # Resize images to target size if needed
        if images[0].shape[0] != img_size:
            from PIL import Image as PILImage
            resized_images = []
            for img in images:
                pil_img = PILImage.fromarray(img)
                pil_img = pil_img.resize((img_size, img_size), PILImage.LANCZOS)
                resized_images.append(np.array(pil_img))
            images = resized_images
        
        os.makedirs(out_dir, exist_ok=True)
        meta = []
        
        for i, img in enumerate(tqdm(images, desc="Saving GAF images")):
            path = os.path.join(out_dir, f"rec_{i}.png")
            save_image(img, path)
            meta.append({'id': i, 'img_path': path, 'label': int(y_bin[i])})
    
    else:
        # Standard quantization methods
        Xq, scaler = normalize_and_quantize(X, n_levels)
        
        if method == 'correlation':
            coords = correlation_based_mapping(X, img_size)
        elif method == 'tsne':
            coords = tsne_grid_encoding(X, img_size)
        else:  # method == 'simple'
            coords = None
        
        os.makedirs(out_dir, exist_ok=True)
        meta = []
        
        for i, row in enumerate(tqdm(Xq, desc=f"Saving {method} images")):
            if coords is not None:
                img = create_structured_image(row, coords, img_size)
            else:
                img = row_to_image(row, img_size)  # Original method
            
            path = os.path.join(out_dir, f"rec_{i}.png")
            save_image(img, path)
            meta.append({'id': i, 'img_path': path, 'label': int(y_bin[i])})
    
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    
    # Save method info
    method_info = {
        'encoding_method': method,
        'img_size': img_size,
        'n_levels': n_levels if method not in ['gaf_sum', 'gaf_diff'] else 'N/A',
        'n_samples': len(meta),
        'n_features': X.shape[1]
    }
    
    import json
    with open(os.path.join(out_dir, 'encoding_info.json'), 'w') as f:
        json.dump(method_info, f, indent=2)
    
    print(f"Generated {len(meta)} images using {method} method")
    return meta_df, method_info


if __name__ == "__main__":
    print("Generating quantized images...")
    build_and_save_images()
