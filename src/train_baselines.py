# src/train_baselines.py
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import CSV_FILE, RANDOM_STATE
from sklearn.preprocessing import LabelEncoder


def run_tabular_baselines(csv_path=CSV_FILE):
    df = pd.read_csv(csv_path, sep=";")  # тут одразу вкажемо правильний роздільник
    if 'target' not in df.columns:
        raise ValueError("CSV must contain 'target' column")

    X = df.drop(columns=['target'])
    y = df['target'].astype(str).str.strip()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # замість y передаємо y_encoded у моделі

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # CatBoost
    cb = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    cb.fit(X_train, y_train)
    preds_cb = cb.predict(X_test)
    print("CatBoost: acc", accuracy_score(y_test, preds_cb),
          "f1_macro", f1_score(y_test, preds_cb, average="macro"))

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    print("XGBoost: acc", accuracy_score(y_test, preds_xgb),
          "f1_macro", f1_score(y_test, preds_xgb, average="macro"))


if __name__=="__main__":
    run_tabular_baselines()
