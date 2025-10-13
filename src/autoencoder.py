# src/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm
from config import DEVICE, BATCH_SIZE, OUT_DIR
import json


class SimpleConvAutoencoder(nn.Module):
    """Simple CNN Autoencoder for image reconstruction"""
    
    def __init__(self, input_channels=1, img_size=32):
        super().__init__()
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Layer 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Layer 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        # Calculate flattened size after encoder
        self.encoded_size = 128 * (img_size // 8) * (img_size // 8)
        
        # Bottleneck (latent space)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoded_size),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        flattened = self.bottleneck(encoded)
        unflattened = flattened.view(-1, 128, self.img_size // 8, self.img_size // 8)
        
        # Decode
        decoded = self.decoder(unflattened)
        
        return decoded
    
    def encode(self, x):
        """Get encoded representation"""
        with torch.no_grad():
            encoded = self.encoder(x)
            latent = self.bottleneck(encoded)
            return latent


class DeepConvAutoencoder(nn.Module):
    """Deeper CNN Autoencoder with skip connections"""
    
    def __init__(self, input_channels=1, img_size=32):
        super().__init__()
        self.img_size = img_size
        
        # Encoder
        self.enc1 = self._make_conv_block(input_channels, 32)
        self.enc2 = self._make_conv_block(32, 64)
        self.enc3 = self._make_conv_block(64, 128)
        self.enc4 = self._make_conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(256, 512)
        
        # Decoder with skip connections
        self.dec4 = self._make_deconv_block(512, 256)
        self.dec3 = self._make_deconv_block(512, 128)  # 256 + 256 from skip
        self.dec2 = self._make_deconv_block(256, 64)   # 128 + 128 from skip
        self.dec1 = self._make_deconv_block(128, 32)   # 64 + 64 from skip
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)      # 32x32 -> 16x16
        e2 = self.enc2(e1)     # 16x16 -> 8x8  
        e3 = self.enc3(e2)     # 8x8 -> 4x4
        e4 = self.enc4(e3)     # 4x4 -> 2x2
        
        # Bottleneck
        b = self.bottleneck(e4)  # 2x2 -> 1x1 -> 2x2
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        
        # Final output
        output = self.final(d1)
        
        return output


def train_autoencoder(model, dataloader, epochs=50, lr=1e-3, device=DEVICE, save_path=None):
    """Train autoencoder"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = []
    best_loss = float('inf')
    
    print(f"Training autoencoder for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        history.append({'epoch': epoch+1, 'loss': avg_loss})
        
        # Save best model
        if avg_loss < best_loss and save_path:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    return history


def evaluate_reconstruction_quality(model, dataloader, device=DEVICE):
    """Evaluate reconstruction quality with various metrics"""
    model.eval()
    
    all_originals = []
    all_reconstructed = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Evaluating reconstruction"):
            images = images.to(device)
            reconstructed = model(images)
            
            # Convert to numpy for metric calculation
            orig_np = images.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()
            
            all_originals.append(orig_np)
            all_reconstructed.append(recon_np)
    
    # Concatenate all batches
    originals = np.concatenate(all_originals, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(originals.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(originals.flatten(), reconstructed.flatten())
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))  # Assuming images are in [0,1] range
    else:
        psnr = float('inf')
    
    # Structural Similarity Index (approximation)
    def ssim_approx(img1, img2):
        """Simple SSIM approximation"""
        mu1, mu2 = np.mean(img1), np.mean(img2)
        sigma1, sigma2 = np.std(img1), np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return ssim
    
    ssim_scores = []
    for i in range(originals.shape[0]):
        ssim_score = ssim_approx(originals[i].flatten(), reconstructed[i].flatten())
        ssim_scores.append(ssim_score)
    
    avg_ssim = np.mean(ssim_scores)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'ssim': avg_ssim,
        'reconstruction_error': np.mean(np.abs(originals - reconstructed))
    }
    
    return metrics, originals, reconstructed


def visualize_reconstructions(originals, reconstructed, save_dir, num_samples=10):
    """Visualize original vs reconstructed images"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(originals), min(num_samples, len(originals)), replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        orig_img = originals[idx].squeeze() if originals[idx].ndim > 2 else originals[idx]
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstructed
        recon_img = reconstructed[idx].squeeze() if reconstructed[idx].ndim > 2 else reconstructed[idx]
        axes[1].imshow(recon_img, cmap='gray')
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        # Difference
        diff_img = np.abs(orig_img - recon_img)
        im = axes[2].imshow(diff_img, cmap='hot')
        axes[2].set_title('Difference')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'reconstruction_{i}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(indices)} reconstruction visualizations to {save_dir}")


def analyze_information_preservation(model, dataloader, original_data, device=DEVICE):
    """Analyze how well information is preserved through the autoencoder"""
    model.eval()
    
    # Extract encoded features
    encoded_features = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            if hasattr(model, 'encode'):
                encoded = model.encode(images).cpu().numpy()
            else:
                # For models without explicit encode method
                encoded = model.encoder(images)
                encoded = model.bottleneck(encoded).cpu().numpy()
            
            encoded_features.append(encoded)
            labels.extend(batch_labels.cpu().numpy())
    
    encoded_features = np.concatenate(encoded_features, axis=0)
    labels = np.array(labels)
    
    # Analyze latent space
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    # PCA on encoded features
    if encoded_features.shape[1] > 2:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(encoded_features)
        explained_variance = pca.explained_variance_ratio_
    else:
        pca_features = encoded_features
        explained_variance = [1.0, 0.0]
    
    # Silhouette score for clustering quality
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(encoded_features, labels)
    else:
        sil_score = 0.0
    
    # Visualization of latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='Set1', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Latent Space (PCA) - Silhouette Score: {sil_score:.3f}')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    
    analysis_results = {
        'encoded_features': encoded_features,
        'pca_features': pca_features,
        'explained_variance': explained_variance,
        'silhouette_score': sil_score,
        'latent_dim': encoded_features.shape[1]
    }
    
    return analysis_results


def comprehensive_autoencoder_analysis(dataloader, img_size=32, output_dir=None):
    """Perform comprehensive autoencoder analysis"""
    if output_dir is None:
        output_dir = os.path.join(OUT_DIR, 'autoencoder_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive autoencoder analysis...")
    
    results = {}
    
    # Test different autoencoder architectures
    models = {
        'simple': SimpleConvAutoencoder(input_channels=1, img_size=img_size),
        'deep': DeepConvAutoencoder(input_channels=1, img_size=img_size)
    }
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} autoencoder ===")
        
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Train model
        save_path = os.path.join(model_dir, f'{model_name}_autoencoder.pth')
        history = train_autoencoder(model, dataloader, epochs=30, save_path=save_path)
        
        # Load best model
        model.load_state_dict(torch.load(save_path))
        
        # Evaluate reconstruction
        metrics, originals, reconstructed = evaluate_reconstruction_quality(model, dataloader)
        
        # Visualize reconstructions
        vis_dir = os.path.join(model_dir, 'visualizations')
        visualize_reconstructions(originals, reconstructed, vis_dir)
        
        # Analyze latent space
        analysis = analyze_information_preservation(model, dataloader, originals)
        
        # Save latent space plot
        plt.savefig(os.path.join(model_dir, 'latent_space.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save results
        model_results = {
            'model_name': model_name,
            'training_history': history,
            'reconstruction_metrics': metrics,
            'latent_analysis': {
                'silhouette_score': analysis['silhouette_score'],
                'explained_variance': analysis['explained_variance'].tolist(),
                'latent_dim': analysis['latent_dim']
            },
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
        # Save to JSON
        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump(model_results, f, indent=2)
        
        results[model_name] = model_results
        
        print(f"Results for {model_name}:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  Silhouette Score: {analysis['silhouette_score']:.4f}")
    
    # Create comparison summary
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'MSE': result['reconstruction_metrics']['mse'],
            'PSNR': result['reconstruction_metrics']['psnr'],
            'SSIM': result['reconstruction_metrics']['ssim'],
            'Silhouette_Score': result['latent_analysis']['silhouette_score'],
            'Parameters': result['model_params'],
            'Final_Loss': result['training_history'][-1]['loss']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'autoencoder_comparison.csv'), index=False)
    
    print("\n=== AUTOENCODER COMPARISON SUMMARY ===")
    print(summary_df.round(6).to_string(index=False))
    
    print(f"\nComprehensive autoencoder analysis completed. Results saved to {output_dir}")
    
    return results, summary_df


if __name__ == "__main__":
    # Example usage
    from dataset_builder import TabImageDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    
    # Load dataset (you'll need to adapt this to your specific dataset)
    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    # dataset = TabImageDataset('path/to/meta.csv', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # comprehensive_autoencoder_analysis(dataloader)
    print("Autoencoder module ready. Import and use comprehensive_autoencoder_analysis() function.")
