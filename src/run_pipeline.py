# src/run_pipeline.py
import os
import time
import pandas as pd
import json
from datetime import datetime
from config import CSV_FILE, IMG_DIR, QUANT_LEVELS, OUT_DIR, IMG_SIZE
from quant_image import build_and_save_images
from train_baselines import run_tabular_baselines
from train_cnn import build_dataloaders, build_model, train_loop
from explainability import comprehensive_model_analysis
from autoencoder import comprehensive_autoencoder_analysis
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def run_comprehensive_experiment(experiment_config):
    """Run comprehensive experiment with all configurations"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(OUT_DIR, f"comprehensive_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Starting comprehensive experiment at {datetime.now()}")
    print(f"Results will be saved to: {experiment_dir}")
    
    # Save experiment configuration
    with open(os.path.join(experiment_dir, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    all_results = []
    
    # 1. Run baseline models once
    print("\n" + "="*50)
    print("STEP 1: BASELINE TABULAR MODELS")
    print("="*50)
    
    baseline_results = run_tabular_baselines(save_results=True)
    
    # 2. Test different image encoding methods and CNN configurations
    encoding_methods = experiment_config.get('encoding_methods', ['simple'])
    cnn_configs = experiment_config.get('cnn_configs', [{'model': 'resnet18', 'pretrained': False, 'freeze': False, 'augmentation': False}])
    
    for encoding_method in encoding_methods:
        print(f"\n" + "="*50)
        print(f"ENCODING METHOD: {encoding_method.upper()}")
        print("="*50)
        
        for img_size in experiment_config.get('img_sizes', [32]):
            for quant_levels in experiment_config.get('quant_levels', [64]):
                
                print(f"\n--- Image size: {img_size}, Quantization: {quant_levels} ---")
                
                # Generate images with current configuration
                img_dir = os.path.join(experiment_dir, f"images_{encoding_method}_s{img_size}_q{quant_levels}")
                meta_df, method_info = build_and_save_images(
                    csv_path=CSV_FILE, 
                    out_dir=img_dir, 
                    img_size=img_size, 
                    n_levels=quant_levels,
                    method=encoding_method
                )
                
                # Build dataloaders
                dl_train, dl_val = build_dataloaders(img_dir, img_size=img_size, use_augmentation=False)
                dl_train_aug, dl_val_aug = build_dataloaders(img_dir, img_size=img_size, use_augmentation=True)
                
                # Test different CNN configurations
                for cnn_config in cnn_configs:
                    config_str = f"{cnn_config['model']}_{'pretrained' if cnn_config['pretrained'] else 'scratch'}_{'frozen' if cnn_config.get('freeze', False) else 'unfrozen'}_{'aug' if cnn_config['augmentation'] else 'noaug'}"
                    
                    print(f"\n  Testing CNN: {config_str}")
                    
                    # Select appropriate dataloader
                    train_dl = dl_train_aug if cnn_config['augmentation'] else dl_train
                    val_dl = dl_val_aug if cnn_config['augmentation'] else dl_val
                    
                    # Build model
                    model = build_model(
                        name=cnn_config['model'], 
                        num_classes=2, 
                        pretrained=cnn_config['pretrained'],
                        freeze_backbone=cnn_config.get('freeze', False)
                    )
                    
                    # Train model
                    model_save_path = os.path.join(experiment_dir, f"model_{encoding_method}_s{img_size}_q{quant_levels}_{config_str}.pth")
                    history, model_info = train_loop(
                        model, train_dl, val_dl, 
                        epochs=experiment_config.get('epochs', 20),
                        save_path=model_save_path,
                        model_name=config_str
                    )
                    
                    # Get best results
                    best_epoch = max(history, key=lambda x: x['acc'])
                    
                    # Record results
                    result = {
                        'encoding_method': encoding_method,
                        'img_size': img_size,
                        'quant_levels': quant_levels,
                        'cnn_model': cnn_config['model'],
                        'pretrained': cnn_config['pretrained'],
                        'frozen_backbone': cnn_config.get('freeze', False),
                        'augmentation': cnn_config['augmentation'],
                        'best_accuracy': best_epoch['acc'],
                        'best_f1_macro': best_epoch['f1_macro'],
                        'best_f1_weighted': best_epoch['f1_weighted'],
                        'best_roc_auc': best_epoch['roc_auc'],
                        'total_params': model_info['total_params'],
                        'trainable_params': model_info['trainable_params'],
                        'flops': model_info['flops'],
                        'total_train_time': model_info['total_train_time'],
                        'avg_epoch_time': model_info['avg_epoch_time']
                    }
                    
                    all_results.append(result)
                    
                    # Run explainability analysis for selected configurations
                    if experiment_config.get('run_explainability', False) and encoding_method == 'simple' and img_size == 32:
                        print(f"    Running explainability analysis for {config_str}...")
                        analysis_dir = os.path.join(experiment_dir, f"explainability_{config_str}")
                        try:
                            comprehensive_model_analysis(model, val_dl, cnn_config['model'], analysis_dir)
                        except Exception as e:
                            print(f"    Explainability analysis failed: {e}")
                
                # Run autoencoder analysis for this image configuration
                if experiment_config.get('run_autoencoder', False) and encoding_method == 'simple':
                    print(f"\n  Running autoencoder analysis for {encoding_method} s{img_size} q{quant_levels}...")
                    try:
                        autoencoder_dir = os.path.join(experiment_dir, f"autoencoder_{encoding_method}_s{img_size}_q{quant_levels}")
                        comprehensive_autoencoder_analysis(dl_train, img_size=img_size, output_dir=autoencoder_dir)
                    except Exception as e:
                        print(f"  Autoencoder analysis failed: {e}")
    
    # 3. Save and analyze all results
    print("\n" + "="*50)
    print("ANALYZING RESULTS")
    print("="*50)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(experiment_dir, 'all_cnn_results.csv'), index=False)
    
    # Create summary visualizations
    create_result_visualizations(results_df, baseline_results, experiment_dir)
    
    # Create comprehensive summary report
    create_summary_report(results_df, baseline_results, experiment_dir, experiment_config)
    
    print(f"\nComprehensive experiment completed! Results saved to: {experiment_dir}")
    return results_df, experiment_dir


def create_result_visualizations(cnn_results, baseline_results, output_dir):
    """Create comprehensive visualizations of results"""
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. CNN vs Baselines comparison
    plt.figure(figsize=(12, 8))
    
    # Get best CNN result for each configuration
    best_cnn = cnn_results.loc[cnn_results['best_accuracy'].idxmax()]
    
    models = list(baseline_results['model']) + [f"CNN-{best_cnn['cnn_model']}"]
    accuracies = list(baseline_results['accuracy']) + [best_cnn['best_accuracy']]
    
    bars = plt.bar(models, accuracies, color=['skyblue']*len(baseline_results) + ['orange'])
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(vis_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. CNN performance by encoding method
    if len(cnn_results['encoding_method'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        encoding_perf = cnn_results.groupby('encoding_method')['best_accuracy'].max()
        bars = plt.bar(encoding_perf.index, encoding_perf.values, color='lightgreen')
        plt.title('Best Performance by Encoding Method')
        plt.ylabel('Best Accuracy')
        plt.xticks(rotation=45)
        
        for bar, acc in zip(bars, encoding_perf.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'encoding_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Parameters vs Performance scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(cnn_results['total_params'], cnn_results['best_accuracy'], 
                         c=cnn_results['total_train_time'], cmap='viridis', alpha=0.7)
    plt.xlabel('Total Parameters')
    plt.ylabel('Best Accuracy')
    plt.title('Parameters vs Performance (color = training time)')
    plt.colorbar(scatter, label='Training Time (s)')
    
    # Add annotations for best models
    top_3 = cnn_results.nlargest(3, 'best_accuracy')
    for idx, row in top_3.iterrows():
        plt.annotate(f"{row['cnn_model']} ({row['encoding_method']})", 
                    (row['total_params'], row['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'params_vs_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of configurations
    if len(cnn_results) > 1:
        pivot_data = cnn_results.pivot_table(
            index=['encoding_method', 'img_size'], 
            columns=['cnn_model', 'pretrained'], 
            values='best_accuracy', 
            aggfunc='max'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Configuration Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'config_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")


def create_summary_report(cnn_results, baseline_results, output_dir, experiment_config):
    """Create comprehensive summary report"""
    
    report_path = os.path.join(output_dir, 'experiment_summary.md')
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Tabular-to-Image CNN Experiment Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Experiment configuration
        f.write("## Experiment Configuration\n\n")
        f.write(f"- **Encoding Methods:** {experiment_config.get('encoding_methods', 'N/A')}\n")
        f.write(f"- **Image Sizes:** {experiment_config.get('img_sizes', 'N/A')}\n")
        f.write(f"- **Quantization Levels:** {experiment_config.get('quant_levels', 'N/A')}\n")
        f.write(f"- **CNN Models:** {[cfg['model'] for cfg in experiment_config.get('cnn_configs', [])]}\n")
        f.write(f"- **Training Epochs:** {experiment_config.get('epochs', 'N/A')}\n\n")
        
        # Baseline results
        f.write("## Baseline Model Results\n\n")
        f.write(baseline_results.to_markdown(index=False))
        f.write("\n\n")
        
        # CNN results summary
        f.write("## CNN Model Results Summary\n\n")
        
        # Best overall model
        best_model = cnn_results.loc[cnn_results['best_accuracy'].idxmax()]
        f.write(f"**Best CNN Model:**\n")
        f.write(f"- Model: {best_model['cnn_model']}\n")
        f.write(f"- Encoding: {best_model['encoding_method']}\n")
        f.write(f"- Pretrained: {best_model['pretrained']}\n")
        f.write(f"- Accuracy: {best_model['best_accuracy']:.4f}\n")
        f.write(f"- F1-Score: {best_model['best_f1_macro']:.4f}\n")
        f.write(f"- ROC-AUC: {best_model.get('best_roc_auc', 'N/A')}\n")
        f.write(f"- Parameters: {best_model['total_params']:,}\n")
        f.write(f"- Training Time: {best_model['total_train_time']:.2f}s\n\n")
        
        # Performance comparison
        best_baseline = baseline_results.loc[baseline_results['accuracy'].idxmax()]
        f.write("## Performance Comparison\n\n")
        f.write(f"- **Best Baseline:** {best_baseline['model']} ({best_baseline['accuracy']:.4f})\n")
        f.write(f"- **Best CNN:** {best_model['cnn_model']} ({best_model['best_accuracy']:.4f})\n")
        improvement = (best_model['best_accuracy'] - best_baseline['accuracy']) / best_baseline['accuracy'] * 100
        f.write(f"- **Improvement:** {improvement:+.2f}%\n\n")
        
        # Top 5 configurations
        f.write("## Top 5 CNN Configurations\n\n")
        top_5 = cnn_results.nlargest(5, 'best_accuracy')[[
            'encoding_method', 'cnn_model', 'pretrained', 'best_accuracy', 
            'best_f1_macro', 'total_params', 'total_train_time'
        ]]
        f.write(top_5.to_markdown(index=False))
        f.write("\n\n")
        
        # Insights and conclusions
        f.write("## Key Insights\n\n")
        
        # Best encoding method
        if len(cnn_results['encoding_method'].unique()) > 1:
            best_encoding = cnn_results.groupby('encoding_method')['best_accuracy'].max().idxmax()
            f.write(f"- **Best Encoding Method:** {best_encoding}\n")
        
        # Pretrained vs from scratch
        if 'pretrained' in cnn_results.columns:
            pretrained_avg = cnn_results[cnn_results['pretrained'] == True]['best_accuracy'].mean()
            scratch_avg = cnn_results[cnn_results['pretrained'] == False]['best_accuracy'].mean()
            f.write(f"- **Pretrained Models:** {pretrained_avg:.4f} average accuracy\n")
            f.write(f"- **From Scratch:** {scratch_avg:.4f} average accuracy\n")
        
        # Efficiency analysis
        efficiency_score = cnn_results['best_accuracy'] / cnn_results['total_params'].apply(lambda x: x/1e6)
        efficient_model = cnn_results.loc[efficiency_score.idxmax()]
        f.write(f"- **Most Efficient:** {efficient_model['cnn_model']} ({efficient_model['best_accuracy']:.4f} accuracy, {efficient_model['total_params']/1e6:.1f}M params)\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- `all_cnn_results.csv` - Detailed results for all CNN experiments\n")
        f.write("- `baseline_results.csv` - Tabular baseline model results\n")
        f.write("- `visualizations/` - Performance comparison charts\n")
        if experiment_config.get('run_explainability', False):
            f.write("- `explainability_*/` - Model interpretability analysis\n")
        if experiment_config.get('run_autoencoder', False):
            f.write("- `autoencoder_*/` - Information preservation analysis\n")
    
    print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Tabular-to-Image CNN Experiment Pipeline')
    parser.add_argument("--step", choices=["all", "images", "baselines", "cnn", "comprehensive"], 
                       default="comprehensive", help="Experiment step to run")
    parser.add_argument("--img_size", type=int, default=32, help="Image size")
    parser.add_argument("--quant", type=int, default=256, help="Quantization levels")
    parser.add_argument("--config", type=str, help="Path to experiment configuration JSON")
    args = parser.parse_args()
    
    if args.step == "comprehensive":
        # Default comprehensive experiment configuration
        if args.config:
            with open(args.config, 'r') as f:
                experiment_config = json.load(f)
        else:
            experiment_config = {
                'encoding_methods': ['simple', 'correlation', 'tsne'],
                'img_sizes': [32, 64],
                'quant_levels': [64, 256],
                'cnn_configs': [
                    {'model': 'resnet18', 'pretrained': False, 'freeze': False, 'augmentation': False},
                    {'model': 'resnet18', 'pretrained': True, 'freeze': False, 'augmentation': False},
                    {'model': 'resnet18', 'pretrained': True, 'freeze': True, 'augmentation': False},
                    {'model': 'mobilenet_v2', 'pretrained': False, 'freeze': False, 'augmentation': False},
                    {'model': 'mobilenet_v2', 'pretrained': True, 'freeze': False, 'augmentation': False},
                    {'model': 'resnet18', 'pretrained': False, 'freeze': False, 'augmentation': True},
                ],
                'epochs': 20,
                'run_explainability': True,
                'run_autoencoder': True
            }
        
        run_comprehensive_experiment(experiment_config)
    
    else:
        # Original simple pipeline
        if args.step in ("all","images"):
            print("Step: build images")
            build_and_save_images(csv_path=CSV_FILE, out_dir=IMG_DIR, img_size=args.img_size, n_levels=args.quant)
        
        if args.step in ("all","baselines"):
            print("Step: run tabular baselines")
            run_tabular_baselines()
        
        if args.step in ("all","cnn"):
            print("Step: train cnn on images")
            dl_train, dl_val = build_dataloaders(IMG_DIR)
            model = build_model(num_classes=2)
            history, model_info = train_loop(model, dl_train, dl_val, model_name="resnet18")

if __name__ == "__main__":
    main()
