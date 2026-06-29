"""
Generate per-attacker risk scores from 4 XGBoost models (SIEM/EDR/NDR/XDR).
Run from dashboard/ directory: python generate_attacker_scores.py
Output: ../data/attacker_scores.json
"""

import sys, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from xdr_dataset import generate_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

FEATURE_SETS = {
    'SIEM': ['failed_logins_count', 'new_ip_flag', 'login_hour'],
    'EDR':  ['files_vs_role_threshold', 'file_access_spike', 'admin_tools_used_count'],
    'NDR':  ['bytes_vs_role_threshold', 'distinct_hosts_accessed', 'dst_ip_category'],
    'XDR':  ['failed_logins_count', 'new_ip_flag', 'login_hour',
              'files_vs_role_threshold', 'file_access_spike', 'admin_tools_used_count',
              'bytes_vs_role_threshold', 'distinct_hosts_accessed', 'dst_ip_category',
              'login_hour_deviation', 'bytes_vs_personal_baseline', 'file_to_transfer_gap_mins',
              'sensitive_file_category_deviation', 'role_based_access_score'],
}

USER_IDS = {
    'attack1_phishing':            'user_ATK_001',
    'attack1_spray':               'user_ATK_002',
    'attack1_credential_stuffing': 'user_ATK_003',
    'attack2_planned_exfil':       'user_INS_001',
    'attack2_opportunistic':       'user_INS_002',
    'attack2_gradual_buildup':     'user_INS_003',
}

TARGET_SUBTYPES = [
    ('attack1_phishing',            'Compromised Account', 'Phishing',            'Attack 1'),
    ('attack1_spray',               'Compromised Account', 'Password Spray',       'Attack 1'),
    ('attack1_credential_stuffing', 'Compromised Account', 'Credential Stuffing',  'Attack 1'),
    ('attack2_planned_exfil',       'Admin Insider',       'Planned Exfiltration', 'Attack 2'),
    ('attack2_opportunistic',       'Admin Insider',       'Opportunistic',        'Attack 2'),
    ('attack2_gradual_buildup',     'Admin Insider',       'Gradual Buildup',      'Attack 2'),
]


def main():
    print("Generating dataset (8000 train / 3200 test, seed=42)...")
    # 11200 total: ~56% normal, ~22% attack1, ~22% attack2
    df = generate_dataset(n_normal=5600, n_attack1=2800, n_attack2=2800, seed=42)
    print(f"  Total: {len(df)} | Normal: {(df.label==0).sum()} | "
          f"Attack1: {(df.label==1).sum()} | Attack2: {(df.label==2).sum()}")

    all_features = FEATURE_SETS['XDR']
    X = df[all_features]
    y = df['label']
    subtypes = df['attack_subtype'].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
        X, y, subtypes, test_size=3200, random_state=42, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    sub_train = sub_train.reset_index(drop=True)
    sub_test  = sub_test.reset_index(drop=True)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining 4 XGBoost models...")
    models = {}
    for name, feats in FEATURE_SETS.items():
        clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, eval_metric='mlogloss',
            verbosity=0
        )
        clf.fit(X_train[feats], y_train)
        models[name] = (clf, feats)
        print(f"  {name:4s}: trained on {len(feats)} features")

    # Build test DataFrame for lookup
    test_df = X_test.copy()
    test_df['label'] = y_test.values
    test_df['attack_subtype'] = sub_test.values

    train_df = X_train.copy()
    train_df['label'] = y_train.values
    train_df['attack_subtype'] = sub_train.values

    print("\nScoring 6 representative attackers...")
    results = []
    for prefix, category, subtype_label, true_label in TARGET_SUBTYPES:
        # Prefer non-stealthy variant for clarity; fall back to any match
        mask_exact   = test_df['attack_subtype'] == prefix
        mask_prefix  = test_df['attack_subtype'].str.startswith(prefix)
        candidates   = test_df[mask_exact] if mask_exact.any() else test_df[mask_prefix]
        if candidates.empty:
            candidates = train_df[train_df['attack_subtype'].str.startswith(prefix)]
        if candidates.empty:
            print(f"  WARNING: no sample found for {prefix}")
            continue

        row = candidates.iloc[0]
        user_id = USER_IDS.get(prefix, f'user_UNK_{prefix[-3:].upper()}')

        scores = {}
        for name, (clf, feats) in models.items():
            sample = row[feats].values.reshape(1, -1)
            proba  = clf.predict_proba(sample)[0]
            # Attack probability = 1 - P(normal class=0)
            attack_prob = float(1.0 - proba[0])
            scores[name] = round(min(attack_prob * 100, 100.0), 1)

        results.append({
            'user_id':     user_id,
            'attack_type': category,
            'subtype':     subtype_label,
            'true_label':  true_label,
            'scores':      scores,
        })
        print(f"  {subtype_label:25s} | SIEM={scores['SIEM']:5.1f}  "
              f"EDR={scores['EDR']:5.1f}  NDR={scores['NDR']:5.1f}  XDR={scores['XDR']:5.1f}")

    out = os.path.join(DATA_DIR, 'attacker_scores.json')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} attacker profiles -> {out}")


if __name__ == '__main__':
    main()
