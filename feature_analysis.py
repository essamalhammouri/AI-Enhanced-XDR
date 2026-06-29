"""
Feature Analysis — Advisory Mode
=================================
Analyzes all 16 features to help decide which to keep, remove, or redefine.

Produces 5 analyses:
  1. Correlation matrix — which features are redundant with each other
  2. Mutual information — which features carry the most signal
  3. Per-class importance — which features detect Attack 1 vs Attack 2
  4. Feature removal impact — how much F1 drops when each is removed
  5. Summary recommendations

No automatic decisions — all information shown to you for manual review.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from xdr_dataset import generate_dataset

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 75)
print("  FEATURE ANALYSIS — Advisory Mode")
print("=" * 75)
print("\nGenerating dataset for analysis...")

# Use bigger dataset for more stable analysis
df = generate_dataset(n_normal=800, n_attack1=400, n_attack2=400, seed=42)
print(f"  Dataset: {len(df)} samples")
print(f"  Normal : {(df['label']==0).sum()}")
print(f"  Attack1: {(df['label']==1).sum()}")
print(f"  Attack2: {(df['label']==2).sum()}")

ALL_FEATURES = [
    'failed_logins_count', 'new_ip_flag', 'login_hour',
    'sensitive_files_accessed', 'files_vs_role_threshold', 'file_access_spike',
    'admin_tools_used_count',
    'bytes_sent', 'bytes_vs_role_threshold', 'dst_ip_suspicious', 'distinct_hosts_accessed',
    'login_hour_deviation', 'bytes_vs_personal_baseline',
    'file_to_transfer_gap_mins', 'chain_detected',
    'sensitive_file_category_deviation',
]

FEATURE_LAYER = {
    'failed_logins_count': 'SIEM', 'new_ip_flag': 'SIEM', 'login_hour': 'SIEM',
    'sensitive_files_accessed': 'EDR', 'files_vs_role_threshold': 'EDR',
    'file_access_spike': 'EDR', 'admin_tools_used_count': 'EDR',
    'bytes_sent': 'NDR', 'bytes_vs_role_threshold': 'NDR',
    'dst_ip_suspicious': 'NDR', 'distinct_hosts_accessed': 'NDR',
    'login_hour_deviation': 'XDR', 'bytes_vs_personal_baseline': 'XDR',
    'file_to_transfer_gap_mins': 'XDR', 'chain_detected': 'XDR',
    'sensitive_file_category_deviation': 'XDR',
}

X = df[ALL_FEATURES]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────
# ANALYSIS 1 — CORRELATION MATRIX
# Which features are redundant with each other?
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ANALYSIS 1: CORRELATION MATRIX")
print("  (features with |r| > 0.7 are REDUNDANT — consider removing one)")
print("=" * 75)

corr = X.corr().abs()

# Find high correlation pairs
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if r > 0.7:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], r))

if high_corr_pairs:
    print("\n  ⚠️  HIGH CORRELATION PAIRS (redundant features):")
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"    {f1:<36} ↔ {f2:<36} r = {r:.3f}")
else:
    print("\n  ✅ No highly correlated feature pairs (all |r| < 0.7)")

# Plot heatmap
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7}, ax=ax)
ax.set_title('Feature Correlation Matrix\n(|r| > 0.7 = redundant)',
             fontweight='bold', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('data/feature_correlation.png', dpi=150)
plt.close()
print(f"\n  📊 Saved: data/feature_correlation.png")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 2 — MUTUAL INFORMATION
# Which features carry the most signal about the label?
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ANALYSIS 2: MUTUAL INFORMATION")
print("  (how much each feature tells us about the label, 0 = nothing)")
print("=" * 75)

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'layer': [FEATURE_LAYER[f] for f in ALL_FEATURES],
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\n  {'Layer':<8}{'Feature':<38} {'MI Score':<12} Signal")
print(f"  {'─'*75}")
for _, row in mi_df.iterrows():
    bar = '█' * int(row['mi_score'] * 40)
    strength = ("STRONG" if row['mi_score'] > 0.3 else
                "MEDIUM" if row['mi_score'] > 0.1 else
                "WEAK  " if row['mi_score'] > 0.03 else
                "NONE  ")
    print(f"  [{row['layer']:<4}] {row['feature']:<35} {row['mi_score']:.4f}      {strength} {bar}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 3 — PER-CLASS IMPORTANCE
# Which features detect Attack 1 vs Attack 2 specifically?
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ANALYSIS 3: PER-CLASS IMPORTANCE")
print("  (which features separate each attack from normal)")
print("=" * 75)

def class_importance(X, y, attack_label):
    """How well does each feature separate attack_label from normal (0)?"""
    mask = (y == 0) | (y == attack_label)
    y_bin = (y[mask] == attack_label).astype(int)
    return mutual_info_classif(X[mask], y_bin, random_state=42)

mi_a1 = class_importance(X, y, 1)
mi_a2 = class_importance(X, y, 2)

per_class = pd.DataFrame({
    'feature': ALL_FEATURES,
    'layer': [FEATURE_LAYER[f] for f in ALL_FEATURES],
    'detects_attack1': mi_a1,
    'detects_attack2': mi_a2
})

# Sort by total usefulness
per_class['total'] = per_class['detects_attack1'] + per_class['detects_attack2']
per_class = per_class.sort_values('total', ascending=False)

print(f"\n  {'Layer':<8}{'Feature':<38} {'Att1':<10}{'Att2':<10}Specialty")
print(f"  {'─'*80}")
for _, row in per_class.iterrows():
    a1, a2 = row['detects_attack1'], row['detects_attack2']
    if a1 > 0.15 and a2 > 0.15:
        specialty = "DETECTS BOTH"
    elif a1 > 0.10 and a2 < 0.05:
        specialty = "Attack1 only"
    elif a2 > 0.10 and a1 < 0.05:
        specialty = "Attack2 only"
    elif a1 < 0.03 and a2 < 0.03:
        specialty = "useless"
    else:
        specialty = "weak"
    print(f"  [{row['layer']:<4}] {row['feature']:<35} {a1:<10.3f}{a2:<10.3f}{specialty}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 4 — FEATURE REMOVAL IMPACT
# Train model with each feature removed, see F1 drop
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ANALYSIS 4: FEATURE REMOVAL IMPACT")
print("  (how much F1 drops when each feature is removed — bigger = more critical)")
print("=" * 75)

RF_PARAMS = {
    'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10,
    'min_samples_leaf': 2, 'max_features': 'sqrt',
    'class_weight': 'balanced', 'random_state': 42
}

# Baseline with ALL features
rf_all = RandomForestClassifier(**RF_PARAMS)
rf_all.fit(X_train, y_train)
baseline_f1 = f1_score(y_test, rf_all.predict(X_test), average='weighted')
print(f"\n  Baseline F1 (all 16 features): {baseline_f1:.4f}\n")

removal_impact = []
for feat in ALL_FEATURES:
    remaining = [f for f in ALL_FEATURES if f != feat]
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train[remaining], y_train)
    f1 = f1_score(y_test, rf.predict(X_test[remaining]), average='weighted')
    removal_impact.append({
        'feature': feat,
        'layer': FEATURE_LAYER[feat],
        'f1_without': f1,
        'drop': baseline_f1 - f1
    })

impact_df = pd.DataFrame(removal_impact).sort_values('drop', ascending=False)

print(f"  {'Layer':<8}{'Feature':<38} {'F1 without':<12}{'Drop':<10}Critical?")
print(f"  {'─'*80}")
for _, row in impact_df.iterrows():
    drop = row['drop']
    if drop > 0.05:   verdict = "🔴 CRITICAL"
    elif drop > 0.02: verdict = "🟡 useful"
    elif drop > 0.005:verdict = "🟢 minor"
    else:             verdict = "⚪ useless/redundant"
    print(f"  [{row['layer']:<4}] {row['feature']:<35} {row['f1_without']:<12.4f}{drop:<10.4f}{verdict}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 5 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ANALYSIS 5: RECOMMENDATIONS FOR YOUR DECISION")
print("=" * 75)

# Categorize features based on combined signals
recommendations = []
for feat in ALL_FEATURES:
    mi = mi_df[mi_df['feature'] == feat]['mi_score'].iloc[0]
    a1 = per_class[per_class['feature'] == feat]['detects_attack1'].iloc[0]
    a2 = per_class[per_class['feature'] == feat]['detects_attack2'].iloc[0]
    drop = impact_df[impact_df['feature'] == feat]['drop'].iloc[0]

    # Check if highly correlated with another feature
    redundant_with = None
    for f1_c, f2_c, r in high_corr_pairs:
        if feat == f1_c:
            redundant_with = f2_c
            break
        elif feat == f2_c:
            redundant_with = f1_c
            break

    # Decision logic
    if drop < 0.002 and mi < 0.03:
        action = "❌ REMOVE"
        reason = "no signal, no impact"
    elif drop < 0.003 and redundant_with:
        action = "❌ REMOVE"
        reason = f"redundant with {redundant_with}"
    elif drop > 0.05:
        action = "✅ KEEP"
        reason = "critical feature"
    elif drop > 0.01:
        action = "✅ KEEP"
        reason = "useful feature"
    elif mi > 0.10:
        action = "⚠️  REVIEW"
        reason = "has signal but low removal impact — maybe redundant"
    else:
        action = "⚠️  REVIEW"
        reason = "weak contribution"

    recommendations.append({
        'feature': feat, 'layer': FEATURE_LAYER[feat],
        'mi': mi, 'a1': a1, 'a2': a2, 'drop': drop,
        'action': action, 'reason': reason
    })

rec_df = pd.DataFrame(recommendations).sort_values('drop', ascending=False)

print(f"\n  {'Layer':<8}{'Feature':<38} {'Action':<14}Reason")
print(f"  {'─'*85}")
for _, row in rec_df.iterrows():
    print(f"  [{row['layer']:<4}] {row['feature']:<35} {row['action']:<14}{row['reason']}")

# ─────────────────────────────────────────────────────────────
# VISUAL: Combined feature scorecard
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Plot 1: MI scores
mi_sorted = mi_df.sort_values('mi_score', ascending=True)
colors_layer = {'SIEM': '#3498db', 'EDR': '#2ecc71', 'NDR': '#e67e22', 'XDR': '#e74c3c'}
axes[0].barh(mi_sorted['feature'],
             mi_sorted['mi_score'],
             color=[colors_layer[l] for l in mi_sorted['layer']],
             edgecolor='black')
axes[0].set_title('Mutual Information (overall)', fontweight='bold')
axes[0].set_xlabel('MI Score')

# Plot 2: Per-class breakdown
pc_sorted = per_class.sort_values('total', ascending=True)
w = 0.4
positions = np.arange(len(pc_sorted))
axes[1].barh(positions - w/2, pc_sorted['detects_attack1'], w,
             label='Attack 1 (Compromised)', color='#f39c12', edgecolor='black')
axes[1].barh(positions + w/2, pc_sorted['detects_attack2'], w,
             label='Attack 2 (Insider)', color='#9b59b6', edgecolor='black')
axes[1].set_yticks(positions)
axes[1].set_yticklabels(pc_sorted['feature'])
axes[1].set_title('Per-Class Feature Importance', fontweight='bold')
axes[1].set_xlabel('MI Score')
axes[1].legend(loc='lower right')

# Plot 3: Removal impact
imp_sorted = impact_df.sort_values('drop', ascending=True)
bar_colors = ['#e74c3c' if d > 0.05 else '#f39c12' if d > 0.02
              else '#2ecc71' if d > 0.005 else '#95a5a6'
              for d in imp_sorted['drop']]
axes[2].barh(imp_sorted['feature'], imp_sorted['drop'],
             color=bar_colors, edgecolor='black')
axes[2].set_title('F1 Drop When Removed', fontweight='bold')
axes[2].set_xlabel('F1 Drop')
axes[2].axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
axes[2].axvline(0.005, color='gray', linestyle='--', alpha=0.5, label='Useless threshold')
axes[2].legend()

plt.tight_layout()
plt.savefig('data/feature_analysis.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
keep = rec_df[rec_df['action'].str.contains('KEEP')]
remove = rec_df[rec_df['action'].str.contains('REMOVE')]
review = rec_df[rec_df['action'].str.contains('REVIEW')]

print("\n" + "=" * 75)
print("  SUMMARY")
print("=" * 75)
print(f"\n  ✅ KEEP  ({len(keep)} features) — proven critical or useful")
for _, row in keep.iterrows():
    print(f"     • {row['feature']:<35} [{row['layer']}]  drop={row['drop']:.4f}")

print(f"\n  ❌ REMOVE ({len(remove)} features) — redundant or useless")
for _, row in remove.iterrows():
    print(f"     • {row['feature']:<35} [{row['layer']}]  {row['reason']}")

print(f"\n  ⚠️  REVIEW ({len(review)} features) — you decide")
for _, row in review.iterrows():
    print(f"     • {row['feature']:<35} [{row['layer']}]  {row['reason']}")

print("\n  📊 Plots saved:")
print("     data/feature_correlation.png")
print("     data/feature_analysis.png")
print("\n" + "=" * 75)
print("  NEXT STEP: Review recommendations, tell me which features to keep")
print("=" * 75)