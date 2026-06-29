"""
XDR ML Model — Final Clean Version (14 features)
Baseline model. Use tune_model.py next for hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from xdr_dataset import generate_dataset

SIEM_FEATURES = ['failed_logins_count', 'new_ip_flag', 'login_hour']
EDR_FEATURES  = ['files_vs_role_threshold', 'file_access_spike', 'admin_tools_used_count']
NDR_FEATURES  = ['bytes_vs_role_threshold', 'distinct_hosts_accessed', 'dst_ip_category']
XDR_ONLY_FEATURES = [
    'login_hour_deviation', 'bytes_vs_personal_baseline',
    'file_to_transfer_gap_mins', 'sensitive_file_category_deviation',
    'role_based_access_score',
]
XDR_FEATURES = SIEM_FEATURES + EDR_FEATURES + NDR_FEATURES + XDR_ONLY_FEATURES

FEATURE_SETS = {
    'SIEM (Identity only)': SIEM_FEATURES,
    'EDR  (Endpoint only)': EDR_FEATURES,
    'NDR  (Network only)':  NDR_FEATURES,
    'XDR  (All layers)':    XDR_FEATURES,
}

TARGET_NAMES = ['Normal', 'Attack1-Compromised', 'Attack2-AdminInsider']

# Default baseline hyperparameters — will tune later
RF_PARAMS = {
    'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10,
    'min_samples_leaf': 2, 'max_features': 'sqrt',
    'class_weight': 'balanced', 'random_state': 42
}

print("=" * 72)
print("  XDR ML MODEL — Final Clean Version (14 features)")
print("=" * 72)

print("\n[Step 1] Generating TRAINING dataset (seed=None)...")
train_df = generate_dataset(n_normal=500, n_attack1=250, n_attack2=250, seed=None)

print("[Step 2] Generating TEST dataset (seed=None, separate)...")
test_df  = generate_dataset(n_normal=200, n_attack1=100, n_attack2=100, seed=None)

print(f"\n  Training: {len(train_df)}  Test: {len(test_df)}")
print(f"  Features: {len(XDR_FEATURES)} (SIEM=3, EDR=3, NDR=3, XDR=5)")

y_train = train_df['label']
y_test  = test_df['label']

train_df.to_csv('data/features.csv', index=False)

# ─────────────────────────────────────────────────────────────
# TEST 1 — ALL 4 MODELS
# ─────────────────────────────────────────────────────────────
results = {}
print("\n" + "=" * 72)
print("  [TEST 1] TRAIN vs INDEPENDENT TEST SET")
print("=" * 72)

for name, feats in FEATURE_SETS.items():
    X_tr, X_te = train_df[feats], test_df[feats]
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_train)

    train_f1 = f1_score(y_train, rf.predict(X_tr), average='weighted')
    y_pred = rf.predict(X_te)
    f1  = f1_score(y_test, y_pred, average='weighted')
    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    a1_recall = (y_pred[y_test == 1] == 1).mean()
    a2_recall = (y_pred[y_test == 2] == 2).mean()

    gap = train_f1 - f1
    flag = "⚠️  overfitting" if gap > 0.10 else "✅ generalizing"

    results[name] = {
        'f1': f1, 'precision': pre, 'recall': rec,
        'train_f1': train_f1, 'gap': gap,
        'a1_recall': a1_recall, 'a2_recall': a2_recall,
        'y_pred': y_pred, 'model': rf, 'feats': feats
    }

    print(f"\n[{name}]")
    print(f"  Features    : {len(feats)}")
    print(f"  Train F1    : {train_f1:.3f}")
    print(f"  Test F1     : {f1:.3f}")
    print(f"  Gap         : {gap:.3f}  {flag}")
    print(f"  Precision   : {pre:.3f}")
    print(f"  Recall      : {rec:.3f}")
    print(f"  Attack1 Rec : {a1_recall:.3f}")
    print(f"  Attack2 Rec : {a2_recall:.3f}")

# ─────────────────────────────────────────────────────────────
# TEST 2 — NOISE ROBUSTNESS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [TEST 2] NOISE ROBUSTNESS")
print("=" * 72)

xdr_model = results['XDR  (All layers)']['model']
X_test_xdr = test_df[XDR_FEATURES].copy().astype(float)

noise_results = []
for noise_level in [0.00, 0.05, 0.10, 0.15, 0.20]:
    X_noisy = X_test_xdr.copy()
    for col in X_noisy.columns:
        X_noisy[col] += np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
    f1_n = f1_score(y_test, xdr_model.predict(X_noisy), average='weighted')
    noise_results.append((noise_level, f1_n))
    print(f"  Noise {int(noise_level*100):>3}% → F1 = {f1_n:.3f}")

# ─────────────────────────────────────────────────────────────
# TEST 3 — LEARNING CURVES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [TEST 3] LEARNING CURVES")
print("=" * 72)

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(**RF_PARAMS),
    train_df[XDR_FEATURES], y_train,
    train_sizes=np.linspace(0.1, 1.0, 8), cv=5,
    scoring='f1_weighted', n_jobs=-1
)
train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)
print(f"  Train F1: {train_mean[-1]:.3f}  |  Validation F1: {val_mean[-1]:.3f}")
print(f"  Gap: {train_mean[-1] - val_mean[-1]:.3f}")

# ─────────────────────────────────────────────────────────────
# TEST 4 — PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [TEST 4] PERMUTATION IMPORTANCE")
print("=" * 72)

perm = permutation_importance(
    xdr_model, X_test_xdr, y_test,
    n_repeats=10, random_state=42, scoring='f1_weighted', n_jobs=-1
)
perm_df = pd.DataFrame({
    'feature': XDR_FEATURES,
    'drop': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('drop', ascending=False)

def layer(f):
    if f in SIEM_FEATURES: return 'SIEM    '
    if f in EDR_FEATURES:  return 'EDR     '
    if f in NDR_FEATURES:  return 'NDR     '
    return 'XDR-only'

print(f"\n  {'Layer':<12}{'Feature':<38}{'Drop':<14}Verdict")
print(f"  {'─'*78}")
for _, row in perm_df.iterrows():
    d = row['drop']
    v = "✅ strongly used" if d > 0.05 else "✅ used" if d > 0.01 else "— minor"
    print(f"  [{layer(row['feature'])}] {row['feature']:<35} {d:.4f} ± {row['std']:.3f}  {v}")

# ─────────────────────────────────────────────────────────────
# TEST 5 — HELD-OUT SUBTYPE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [TEST 5] HELD-OUT ATTACK SUBTYPE")
print("=" * 72)

train_ho = generate_dataset(500, 250, 250, seed=None,
    a1_methods=['phishing', 'spray'],
    a2_types=['planned_exfil', 'opportunistic'])
test_ho = generate_dataset(200, 100, 100, seed=None,
    a1_methods=['credential_stuffing'],
    a2_types=['gradual_buildup'])

rf_ho = RandomForestClassifier(**RF_PARAMS)
rf_ho.fit(train_ho[XDR_FEATURES], train_ho['label'])
y_pred_ho = rf_ho.predict(test_ho[XDR_FEATURES])
f1_ho = f1_score(test_ho['label'], y_pred_ho, average='weighted')
a1_rec_ho = (y_pred_ho[test_ho['label'] == 1] == 1).mean()
a2_rec_ho = (y_pred_ho[test_ho['label'] == 2] == 2).mean()

print(f"  Overall F1: {f1_ho:.3f}")
print(f"  credential_stuffing recall: {a1_rec_ho:.3f}")
print(f"  gradual_buildup recall    : {a2_rec_ho:.3f}")

# ─────────────────────────────────────────────────────────────
# TEST 6 — CROSS VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [TEST 6] CROSS VALIDATION")
print("=" * 72)

cv_scores = cross_val_score(
    RandomForestClassifier(**RF_PARAMS),
    train_df[XDR_FEATURES], y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted'
)
print(f"  Fold scores: {[round(s,3) for s in cv_scores]}")
print(f"  Mean F1: {cv_scores.mean():.3f}  |  Std: {cv_scores.std():.3f}")

# ─────────────────────────────────────────────────────────────
# ISOLATION FOREST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  ISOLATION FOREST BASELINE")
print("=" * 72)

iso = IsolationForest(n_estimators=100, contamination=0.4, random_state=42)
iso.fit(train_df[train_df['label']==0][XDR_FEATURES])
iso_pred = iso.predict(test_df[XDR_FEATURES])
iso_bin = [0 if p == 1 else 1 for p in iso_pred]
y_bin = [0 if l == 0 else 1 for l in y_test]
iso_f1 = f1_score(y_bin, iso_bin, average='weighted')
print(f"  F1: {iso_f1:.3f}")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  BASELINE RESULTS — ready for hyperparameter tuning next")
print("=" * 72)
print(f"\n  {'Model':<28}{'Train':<8}{'Test':<8}{'Gap':<7}{'Att1Rec':<9}{'Att2Rec':<9}")
print(f"  {'─'*72}")
for name, res in results.items():
    print(f"  {name:<28}{res['train_f1']:<8.3f}{res['f1']:<8.3f}"
          f"{res['gap']:<7.3f}{res['a1_recall']:<9.3f}{res['a2_recall']:<9.3f}")
print(f"  {'Isolation Forest':<28}{'—':<8}{iso_f1:<8.3f}{'—':<7}{'—':<9}{'—':<9}")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print("\n  Generating plots...")

# F1 comparison
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
f1s = [results[n]['f1'] for n in names]
colors = ['#3498db']*3 + ['#e74c3c']
bars = ax.bar(names, f1s, color=colors, edgecolor='black', width=0.5)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('F1 Score')
ax.set_title('Baseline Results — Single Layer vs XDR', fontweight='bold')
plt.xticks(fontsize=9); plt.tight_layout()
plt.savefig('data/f1_comparison.png', dpi=150); plt.close()

# Per-attack recall
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names)); w = 0.35
a1 = [results[n]['a1_recall'] for n in names]
a2 = [results[n]['a2_recall'] for n in names]
ax.bar(x - w/2, a1, w, label='Attack1 (Compromised)', color='#f39c12', edgecolor='black')
ax.bar(x + w/2, a2, w, label='Attack2 (Admin Insider)', color='#9b59b6', edgecolor='black')
for i, (va, vb) in enumerate(zip(a1, a2)):
    ax.text(i - w/2, va + 0.01, f'{va:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, vb + 0.01, f'{vb:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 1.15); ax.set_ylabel('Recall')
ax.set_title('Per-Attack Recall', fontweight='bold'); ax.legend()
plt.tight_layout(); plt.savefig('data/per_attack_recall.png', dpi=150); plt.close()

# Permutation importance
fig, ax = plt.subplots(figsize=(10, 7))
perm_sorted = perm_df.sort_values('drop', ascending=True)
cmap = {f: ('#3498db' if f in SIEM_FEATURES else '#2ecc71' if f in EDR_FEATURES
            else '#e67e22' if f in NDR_FEATURES else '#e74c3c') for f in XDR_FEATURES}
ax.barh(perm_sorted['feature'], perm_sorted['drop'],
        xerr=perm_sorted['std'], color=[cmap[f] for f in perm_sorted['feature']],
        edgecolor='black')
ax.set_xlabel('F1 drop when shuffled')
ax.set_title('Permutation Importance', fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#3498db', label='SIEM'),
    Patch(color='#2ecc71', label='EDR'),
    Patch(color='#e67e22', label='NDR'),
    Patch(color='#e74c3c', label='XDR-only'),
], loc='lower right')
plt.tight_layout(); plt.savefig('data/permutation_importance.png', dpi=150); plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name}\nF1 = {res["f1"]:.3f}', fontweight='bold')
    ax.set_xticklabels(TARGET_NAMES, rotation=15, fontsize=8)
    ax.set_yticklabels(TARGET_NAMES, rotation=0, fontsize=8)
plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('data/confusion_matrices.png', dpi=150); plt.close()

# Learning curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, 'o-', color='#e74c3c', label='Training', linewidth=2)
ax.plot(train_sizes, val_mean, 'o-', color='#3498db', label='Validation', linewidth=2)
ax.fill_between(train_sizes, train_mean - train_scores.std(axis=1),
                train_mean + train_scores.std(axis=1), alpha=0.15, color='#e74c3c')
ax.fill_between(train_sizes, val_mean - val_scores.std(axis=1),
                val_mean + val_scores.std(axis=1), alpha=0.15, color='#3498db')
ax.set_xlabel('Training size'); ax.set_ylabel('F1')
ax.set_title('Learning Curve', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.05)
plt.tight_layout(); plt.savefig('data/learning_curve.png', dpi=150); plt.close()

# Noise robustness
fig, ax = plt.subplots(figsize=(8, 5))
levels = [n[0]*100 for n in noise_results]
scores = [n[1] for n in noise_results]
ax.plot(levels, scores, 'o-', color='#27ae60', linewidth=2, markersize=9)
for lv, sc in zip(levels, scores):
    ax.annotate(f'{sc:.3f}', xy=(lv, sc), xytext=(0, 10),
                textcoords='offset points', ha='center', fontweight='bold')
ax.set_xlabel('Noise %'); ax.set_ylabel('F1')
ax.set_title('Noise Robustness', fontweight='bold')
ax.set_ylim(0.5, 1.05); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('data/noise_robustness.png', dpi=150); plt.close()

# Cross validation
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), cv_scores, color='#3498db', edgecolor='black')
ax.axhline(cv_scores.mean(), color='red', linestyle='--',
           label=f'Mean = {cv_scores.mean():.3f}')
ax.set_xlabel('Fold'); ax.set_ylabel('F1')
ax.set_title('Cross Validation', fontweight='bold')
ax.set_ylim(0.5, 1.05); ax.legend()
plt.tight_layout(); plt.savefig('data/cross_validation.png', dpi=150); plt.close()

print("\n✅ All plots saved to data/")
print("\nNext step: run tune_model.py to optimize hyperparameters")