"""
hyperparameter_tuning.py
EcoPredict Carbon - lightweight GridSearchCV tuning.

Chạy:
    python hyperparameter_tuning.py
"""
from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

from carbon_utils import (
    RANDOM_STATE,
    TARGET_COL,
    FEATURE_COLS,
    make_clf_pipeline,
    make_reg_pipeline,
    evaluate_classifier,
    evaluate_regressor,
)
from imbalance_handler import class_distribution

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)


def balanced_sample(df: pd.DataFrame, label_col: str, max_per_class: int = 450) -> pd.DataFrame:
    """Lấy mẫu cân bằng để GridSearchCV chạy nhanh nhưng vẫn có đủ Low/Medium/High."""
    parts = []
    for _, part in df.groupby(label_col):
        n = min(len(part), max_per_class)
        parts.append(part.sample(n=n, random_state=RANDOM_STATE))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    return out.reset_index(drop=True)


def main() -> None:
    print("=" * 80)
    print("ECOPREDICT CARBON - LIGHTWEIGHT HYPERPARAMETER TUNING")
    print("=" * 80)

    package_path = Path("ecopredict_model_package.joblib")
    if not package_path.exists():
        package_path = Path("outputs/models/ecopredict_model_package.joblib")
    if not package_path.exists():
        raise FileNotFoundError("Không tìm thấy ecopredict_model_package.joblib")

    pkg = joblib.load(package_path)
    meta = pkg.get("metadata", {})
    feature_cols = meta.get("feature_cols", FEATURE_COLS)

    train_df = pkg.get("train_data")
    test_df = pkg.get("test_data")
    if train_df is None or test_df is None:
        raise ValueError("Model package thiếu train_data/test_data")

    train_df = train_df.dropna(subset=[TARGET_COL, "carbon_label_num"]).copy()
    test_df = test_df.dropna(subset=[TARGET_COL, "carbon_label_num"]).copy()

    # Giới hạn mẫu để tuning có thể chạy nhanh trên laptop và repo demo.
    train_small = balanced_sample(train_df, "carbon_label_num", max_per_class=80)
    test_small = balanced_sample(test_df, "carbon_label_num", max_per_class=50)

    X_train = train_small[feature_cols]
    y_train = train_small["carbon_label_num"].astype(int).values
    yreg_train = train_small[TARGET_COL].astype(float).values

    X_test = test_small[feature_cols]
    y_test = test_small["carbon_label_num"].astype(int).values
    yreg_test = test_small[TARGET_COL].astype(float).values

    rows: list[dict[str, object]] = []

    dist = class_distribution(y_train)
    min_class_count = min(dist.values()) if dist else 0
    use_smote = False  # tuning demo chạy nhanh; core training vẫn dùng SMOTE/class_weight
    smote_k = max(1, min(3, min_class_count - 1)) if min_class_count >= 2 else 1

    # 1) Classification default
    default_clf = make_clf_pipeline(
        RandomForestClassifier(
            n_estimators=25,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        use_smote=use_smote,
        smote_k_neighbors=smote_k,
    )
    default_clf.fit(X_train, y_train)
    rows.append({
        "task": "classification",
        "model": "Random Forest default",
        "cv_best_score": np.nan,
        "best_params": "default",
        **evaluate_classifier(default_clf, X_test, y_test),
    })

    # 2) Classification tuned - grid nhỏ để chạy nhanh.
    tuned_clf_pipe = make_clf_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        use_smote=use_smote,
        smote_k_neighbors=smote_k,
    )
    clf_grid = {
        "model__n_estimators": [25, 40],
        "model__max_depth": [8],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }
    clf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    clf_search = GridSearchCV(
        tuned_clf_pipe,
        clf_grid,
        cv=clf_cv,
        scoring="f1_macro",
        n_jobs=1,
        return_train_score=False,
    )
    clf_search.fit(X_train, y_train)
    rows.append({
        "task": "classification",
        "model": "Random Forest tuned GridSearchCV",
        "cv_best_score": float(clf_search.best_score_),
        "best_params": str(clf_search.best_params_),
        **evaluate_classifier(clf_search.best_estimator_, X_test, y_test),
    })

    # 3) Regression default
    default_reg = make_reg_pipeline(
        TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=25,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    )
    default_reg.fit(X_train, yreg_train)
    rows.append({
        "task": "regression",
        "model": "Random Forest regressor default",
        "cv_best_score": np.nan,
        "best_params": "default",
        **evaluate_regressor(default_reg, X_test, yreg_test),
    })

    # 4) Regression tuned - grid nhỏ.
    tuned_reg_pipe = make_reg_pipeline(
        TransformedTargetRegressor(
            regressor=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    )
    reg_grid = {
        "model__regressor__n_estimators": [25, 40],
        "model__regressor__max_depth": [8],
        "model__regressor__min_samples_leaf": [1, 2],
        "model__regressor__max_features": ["sqrt"],
    }
    reg_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    reg_search = GridSearchCV(
        tuned_reg_pipe,
        reg_grid,
        cv=reg_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        return_train_score=False,
    )
    reg_search.fit(X_train, yreg_train)
    rows.append({
        "task": "regression",
        "model": "Random Forest regressor tuned GridSearchCV",
        "cv_best_score": float(reg_search.best_score_),
        "best_params": str(reg_search.best_params_),
        **evaluate_regressor(reg_search.best_estimator_, X_test, yreg_test),
    })

    out = pd.DataFrame(rows)
    out.insert(0, "train_sample_rows", len(train_small))
    out.insert(1, "test_sample_rows", len(test_small))
    out.to_csv(TAB / "hyperparameter_tuning_results.csv", index=False)

    # Biểu đồ so sánh: classification dùng F1-macro, regression dùng R2 nếu có.
    plot_rows = []
    for _, row in out.iterrows():
        if row["task"] == "classification":
            score = row.get("f1_macro", np.nan)
            xlabel = "F1-macro"
        else:
            score = row.get("r2", np.nan)
            xlabel = "R²"
        plot_rows.append({"model": row["model"], "score": score, "metric": xlabel})
    plot_df = pd.DataFrame(plot_rows)
    plt.figure(figsize=(10, 5.5))
    labels = plot_df["model"].astype(str).str.replace("Random Forest", "RF", regex=False)
    plt.barh(labels, plot_df["score"], color="#047857")
    for i, v in enumerate(plot_df["score"].astype(float)):
        if np.isfinite(v):
            plt.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    plt.title("Default vs GridSearchCV tuned models")
    plt.xlabel("F1-macro cho phân loại / R² cho hồi quy")
    plt.tight_layout()
    plt.savefig(FIG / "hyperparameter_tuning_comparison.png", bbox_inches="tight", dpi=180)
    plt.close()

    print(out.to_string(index=False))
    print("Saved tuning outputs to outputs/tables and outputs/figures")


if __name__ == "__main__":
    main()
