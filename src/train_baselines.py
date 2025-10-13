# src/train_baselines.py
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from config import CSV_FILE, RANDOM_STATE, OUT_DIR
from sklearn.preprocessing import LabelEncoder
import os
import json


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scale_data=False):
    """Train and evaluate a single model, returning metrics and timing info"""
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Time training
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    preds = model.predict(X_test_scaled)
    inference_time = time.time() - start_time
    
    # Get probabilities for AUC
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_test)) == 2 else None
    else:
        proba = None
    
    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_weighted = f1_score(y_test, preds, average='weighted')
    
    if proba is not None and len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, proba)
    else:
        roc_auc = None
    
    results = {
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'inference_time': inference_time,
        'n_params': getattr(model, 'n_estimators', 'N/A') if hasattr(model, 'n_estimators') else 'N/A'
    }
    
    return results


def run_tabular_baselines(csv_path=CSV_FILE, save_results=True):
    """Run comprehensive baseline comparison on tabular data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep=";")
    if 'target' not in df.columns:
        raise ValueError("CSV must contain 'target' column")

    X = df.drop(columns=['target'])
    y = df['target'].astype(str).str.strip()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    print(f"Dataset shape: {X.shape}, Target distribution: {np.bincount(y_encoded)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # Define models to test
    models = {
        'CatBoost': CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, iterations=100),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_estimators=100),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100, verbose=-1),
        'Logistic_Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    }
    
    # Store results
    all_results = []
    
    print("\n=== Training Baseline Models ===")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        scale_data = model_name == 'Logistic_Regression'  # Only scale for LogReg
        
        try:
            results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scale_data)
            all_results.append(results)
            
            # Print results
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Macro: {results['f1_macro']:.4f}")
            print(f"  F1-Weighted: {results['f1_weighted']:.4f}")
            if results['roc_auc'] is not None:
                print(f"  ROC-AUC: {results['roc_auc']:.4f}")
            print(f"  Training Time: {results['training_time']:.3f}s")
            print(f"  Inference Time: {results['inference_time']:.3f}s")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            continue
    
    # Save results
    if save_results and all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(OUT_DIR, 'baseline_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Also save as JSON for easier reading
        json_path = os.path.join(OUT_DIR, 'baseline_results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results also saved as JSON: {json_path}")
        
        # Print summary table
        print("\n=== BASELINE RESULTS SUMMARY ===")
        print(results_df.round(4).to_string(index=False))
        
        return results_df
    
    return all_results


if __name__=="__main__":
    run_tabular_baselines()
