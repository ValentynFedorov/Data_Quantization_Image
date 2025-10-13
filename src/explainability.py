# src/explainability.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import cv2
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd


class GradCAM:
    """Grad-CAM implementation for CNN interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam / torch.max(cam)
        
        return cam.squeeze().cpu().numpy()


def get_target_layer(model, model_name):
    """Get the target layer for Grad-CAM based on model architecture"""
    if model_name == "resnet18":
        return model.layer4[-1].conv2
    elif model_name == "mobilenet_v2":
        return model.features[-1][0]
    elif model_name == "efficientnet_b0":
        return model.features[-1][0]
    else:
        # Try to find the last convolutional layer
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        if conv_layers:
            return conv_layers[-1][1]
        else:
            raise ValueError(f"Could not find suitable layer for {model_name}")


def visualize_gradcam(model, dataloader, model_name, save_dir, num_samples=10):
    """Generate and save Grad-CAM visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    
    sample_count = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        for i in range(images.size(0)):
            if sample_count >= num_samples:
                break
                
            image = images[i:i+1]  # Keep batch dimension
            label = labels[i].item()
            
            # Generate CAM
            cam = grad_cam.generate_cam(image)
            
            # Original image
            orig_img = image.squeeze().cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(orig_img, cmap='gray')
            axes[0].set_title(f'Original (Label: {label})')
            axes[0].axis('off')
            
            # CAM heatmap
            axes[1].imshow(cam, cmap='jet', alpha=0.8)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(orig_img, cmap='gray', alpha=0.7)
            axes[2].imshow(cam, cmap='jet', alpha=0.3)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gradcam_sample_{sample_count}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"Generated {sample_count} Grad-CAM visualizations in {save_dir}")


def extract_features(model, dataloader, layer_name='before_fc'):
    """Extract features from a specific layer for dimensionality reduction"""
    model.eval()
    features = []
    labels = []
    
    # Hook to extract features
    def hook_fn(module, input, output):
        features.append(output.cpu().numpy())
    
    # Register hook on the layer before final classification
    if hasattr(model, 'fc'):  # ResNet-like models
        hook = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):  # MobileNet/EfficientNet-like models
        # Find the global average pooling layer
        for name, module in model.named_modules():
            if 'pool' in name.lower() or 'avgpool' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                break
        else:
            # Fallback to the layer before classifier
            modules = list(model.children())
            hook = modules[-2].register_forward_hook(hook_fn)
    else:
        # Generic approach - use second to last layer
        modules = list(model.children())
        hook = modules[-2].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            _ = model(images)
            labels.extend(batch_labels.cpu().numpy())
    
    hook.remove()
    
    # Concatenate all features
    features = np.concatenate(features, axis=0)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    return features, np.array(labels)


def visualize_feature_space(features, labels, method='tsne', save_path=None):
    """Visualize feature space using t-SNE or UMAP"""
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")
    
    print(f"Computing {method.upper()} embedding...")
    embedding = reducer.fit_transform(features)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Set1', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} Visualization of Feature Space')
    plt.xlabel(f'{method.upper()}_1')
    plt.ylabel(f'{method.upper()}_2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature space visualization saved to {save_path}")
    
    return embedding


def analyze_feature_distributions(features, labels, save_dir):
    """Analyze and visualize feature distributions per class"""
    os.makedirs(save_dir, exist_ok=True)
    
    unique_labels = np.unique(labels)
    
    # Feature statistics per class
    stats = []
    for label in unique_labels:
        class_features = features[labels == label]
        stats.append({
            'class': label,
            'mean_activation': np.mean(class_features),
            'std_activation': np.std(class_features),
            'max_activation': np.max(class_features),
            'min_activation': np.min(class_features),
            'n_samples': len(class_features)
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(save_dir, 'feature_stats_by_class.csv'), index=False)
    
    # Plot feature distribution histograms
    plt.figure(figsize=(12, 6))
    for label in unique_labels:
        class_features = features[labels == label]
        plt.hist(class_features.flatten(), alpha=0.5, label=f'Class {label}', bins=50)
    
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title('Feature Value Distributions by Class')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature analysis saved to {save_dir}")
    return stats_df


def create_confusion_heatmap(y_true, y_pred, class_names=None, save_path=None):
    """Create and save confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return cm


def comprehensive_model_analysis(model, dataloader, model_name, output_dir, num_gradcam_samples=10):
    """Perform comprehensive analysis of a trained model"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting comprehensive analysis for {model_name}...")
    
    # 1. Grad-CAM visualization
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    visualize_gradcam(model, dataloader, model_name, gradcam_dir, num_gradcam_samples)
    
    # 2. Feature extraction and dimensionality reduction
    print("Extracting features...")
    features, labels = extract_features(model, dataloader)
    
    # 3. t-SNE visualization
    tsne_path = os.path.join(output_dir, 'tsne_features.png')
    tsne_embedding = visualize_feature_space(features, labels, method='tsne', save_path=tsne_path)
    
    # 4. UMAP visualization
    try:
        umap_path = os.path.join(output_dir, 'umap_features.png')
        umap_embedding = visualize_feature_space(features, labels, method='umap', save_path=umap_path)
    except ImportError:
        print("UMAP not available, skipping UMAP visualization")
        umap_embedding = None
    
    # 5. Feature analysis
    feature_dir = os.path.join(output_dir, 'feature_analysis')
    stats_df = analyze_feature_distributions(features, labels, feature_dir)
    
    # 6. Model predictions for confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # 7. Confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    cm = create_confusion_heatmap(all_labels, all_preds, 
                                  class_names=[f'Class {i}' for i in np.unique(all_labels)],
                                  save_path=cm_path)
    
    print(f"Comprehensive analysis completed. Results saved to {output_dir}")
    
    return {
        'features': features,
        'labels': labels,
        'tsne_embedding': tsne_embedding,
        'umap_embedding': umap_embedding,
        'stats_df': stats_df,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'ground_truth': all_labels
    }