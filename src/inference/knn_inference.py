import config
from src.subtyping.slide_utils import get_oncotree_code
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

def get_subtype(cancer_type, sample_id):
    sample_id = sample_id[:12]
    subtypes = config.CANCER_SUBTYPE_MAP.get(cancer_type)
    sample_subtype = get_oncotree_code(sample_id)
    if sample_subtype in subtypes:
        return subtypes.index(sample_subtype)
    return None

def load_embeddings_and_labels(folder_path, cancer_type, mode):
    X, y, sample_ids = [], [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.npy') and not fname.endswith('.npz'):
            continue
        sample_id = fname[:12]
        label = get_subtype(cancer_type, sample_id)
        if label is None:
            continue
        fpath = os.path.join(folder_path, fname)
        if (mode == "roi" or mode == "gpt_baseline") and fname.endswith('.npy'):
            emb = np.load(fpath, allow_pickle=True)
        elif mode == "tiles" and fname.endswith('.npz'):
            npz = np.load(fpath)
            if "embedding" not in npz:
                continue
            emb = npz["embedding"]
        else:
            continue  # Skip mismatched file types
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        elif emb.ndim != 1:
            print(f"[WARNING] Unexpected shape {emb.shape} in file: {fname}")
            continue
        X.append(emb)
        y.append(label)
        sample_ids.append(sample_id)
    return np.array(X), np.array(y), sample_ids

def evaluate_knn_classifier(X, y, cancer_type, k=5, n_splits=5):
    skf = safe_stratified_kfold(y, n_splits)
    accs, f1s, aucs = [], [], []
    oof_proba = np.zeros((len(y), len(np.unique(y))), dtype=float)
    oof_y = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        # Compute AUROC
        n_classes = len(np.unique(y))
        y_score = clf.predict_proba(X_test)
        oof_proba[test_idx] = y_score
        oof_y[test_idx] = y_test
        if n_classes == 2:
            if len(np.unique(y_test)) >= 2:
                aucs.append(roc_auc_score(y_test, y_score[:, 1]))
        else:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            try:
                auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
                aucs.append(auc)
            except:
                pass  # some folds may not have all classes
    # print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    # print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if aucs:
        # print(f"AUROC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"{cancer_type} AUROC: {np.mean(aucs):.4f}")
    else:
        print("{cancer_type} AUROC not computed (requires binary classification)")
    
    return oof_y, oof_proba

def safe_stratified_kfold(y, n_splits):
    counts = np.bincount(y)
    min_count = counts[counts > 0].min()
    return StratifiedKFold(n_splits=min(n_splits, max(2, min_count)),
                           shuffle=True, random_state=42)

def evaluate_logistic_classifier(X, y, cancer_type, n_splits=5):
    skf = safe_stratified_kfold(y, n_splits)
    accs, f1s, aucs = [], [], []
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(y), n_classes), dtype=float)
    oof_y = np.zeros(len(y), dtype=int)
    n_classes = len(np.unique(y))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        # AUROC for binary or multi-class (macro average)
        y_score = clf.predict_proba(X_test)
        oof_proba[test_idx] = y_score
        oof_y[test_idx] = y_test

        if len(np.unique(y_test)) >= 2:
            if n_classes == 2:
                aucs.append(roc_auc_score(y_test, y_score[:, 1]))
            else:
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                try:
                    auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
                    aucs.append(auc)
                except:
                    pass
    # print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    # print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if aucs:
        # print(f"AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"{cancer_type} AUROC: {np.mean(aucs):.4f}")
    else:
        print("{cancer_type} AUROC not computed (class coverage insufficient)")
    return oof_y, oof_proba

def bootstrap_macro_auroc(y_true, proba, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba  = np.asarray(proba)
    N = len(y_true)
    classes = np.unique(y_true)
    n_classes = len(classes)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)  # bootstrap indices with replacement
        y_b  = y_true[idx]
        p_b  = proba[idx]
        if n_classes == 2:
            try:
                auc = roc_auc_score(y_b, p_b[:, 1])
            except Exception:
                continue
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_b, classes=range(n_classes))
            try:
                auc = roc_auc_score(y_bin, p_b, average='macro', multi_class='ovr')
            except Exception:
                continue
        vals.append(float(auc))
    return np.array(vals)

def crossval_macro_auroc(X, y, clf_type="knn", k=10, n_splits=5):
    skf = safe_stratified_kfold(y, n_splits)
    n_classes = len(np.unique(y))
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if clf_type == "knn":
            clf = KNeighborsClassifier(n_neighbors=k)
        else:  # 'log'
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)
        if len(np.unique(y_te)) < 2:
            continue
        if n_classes == 2:
            aucs.append(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            y_te_bin = label_binarize(y_te, classes=range(n_classes))
            try:
                aucs.append(roc_auc_score(y_te_bin, y_prob, average='macro', multi_class='ovr'))
            except Exception:
                pass
    return float(np.mean(aucs)) if len(aucs) else np.nan

if __name__ == "__main__":
    cancer_types = ["BRCA", "COLON", "LUNG", "RCC", "GLIOMA", "HEP", "ESO",
                "ADREN", "CERVIX", "PLEURA", "SOFT", "TESTIS", "UTERUS"]
    for cancer_type in cancer_types:
        # cancer_type = "BRCA"
        mode = "gpt_baseline" # roi / gpt_baseline / tiles
        encoder = "H-optimus-0" # gigapath / UNI / H-optimus-0
        model = "knn" # knn / log
        folder_path = os.path.join("inference_output", mode, cancer_type, encoder)
        X, y, ids = load_embeddings_and_labels(folder_path, cancer_type, mode)
        print(f"Loaded {len(X)} samples.")
        if model == "knn":
            evaluate_knn_classifier(X, y, cancer_type, k=10, n_splits=5)
        elif model == "log":
            evaluate_logistic_classifier(X, y, cancer_type, n_splits=5)

