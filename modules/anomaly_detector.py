import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# -- WHY THIS FILE EXISTS ------------------------------------------------------
# ml_model.py uses SUPERVISED learning (Random Forest) -- it needs labeled data
# (attacker = 1, normal = 0) to train on. That works well for known attack patterns.
#
# anomaly_detector.py uses UNSUPERVISED learning (Isolation Forest) -- it needs
# NO labels. It just looks for users who behave very differently from everyone else.
#
# WHY THIS MATTERS FOR YOUR PROJECT:
# In the real world, you don't always have labeled attack data. Unsupervised
# anomaly detection can catch NEW attack patterns that have never been seen before.
# Having both approaches in your XDR system makes it stronger and more realistic.
# In your report, present this as: "supervised detection for known patterns +
# unsupervised detection for unknown/novel threats."

# -- Load features -------------------------------------------------------------
features = pd.read_csv('data/features.csv')

# -- Feature columns for anomaly detection ------------------------------------
# WHY these 14 columns: We use all behavioral features but exclude 'user' (it's
# a name, not a number) and 'label' (Isolation Forest is unsupervised -- it must
# NOT see the labels, otherwise we're cheating).
FEATURE_COLS = [
    'total_logins', 'failed_logins', 'login_failure_rate',
    'after_hours', 'unique_locations',
    'file_access_count', 'usb_events', 'file_copies',
    'admin_actions', 'sensitive_files',
    'bytes_sent', 'unique_destinations',
    'outbound_connections', 'suspicious_conns',
]

X = features[FEATURE_COLS].fillna(0)
# WHY fillna(0): If a user has no network activity, their bytes_sent is NaN.
# We treat missing data as zero activity, which is the correct assumption here.

# -- Standardize features ------------------------------------------------------
# WHY StandardScaler: bytes_sent can be in the millions while login_failure_rate
# is between 0 and 1. Without scaling, Isolation Forest would treat bytes_sent
# as far more important just because its numbers are larger. Scaling puts all
# features on the same numerical range so each contributes equally.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- Run Isolation Forest ------------------------------------------------------
# WHY contamination=0.02: This tells the model to expect about 2% of users to
# be anomalous. We have 9 attackers out of ~500 users = ~1.8%. Setting this to
# 0.02 is an honest estimate. The old code used 0.05 (5%) which would flag
# ~25 users as threats -- too many false alarms.
print("Running Isolation Forest anomaly detection...")
model = IsolationForest(contamination=0.02, random_state=42)
features['anomaly_score_raw'] = model.fit_predict(X_scaled)

# WHY score_samples: fit_predict gives -1 (anomaly) or 1 (normal).
# score_samples gives a continuous score -- lower = more anomalous.
# We negate it so HIGHER score = MORE suspicious (more intuitive).
features['anomaly_score'] = -model.score_samples(X_scaled)

# -- Normalize risk score to 0-100 --------------------------------------------
# WHY: The raw anomaly score is hard to interpret (e.g. 0.58 vs 0.61 -- what does
# that mean?). Mapping it to 0-100 makes it immediately understandable for your
# report and for the correlation engine. A score of 87 clearly means "high risk."
score_min = features['anomaly_score'].min()
score_max = features['anomaly_score'].max()
features['risk_score_0_100'] = (
    (features['anomaly_score'] - score_min) / (score_max - score_min) * 100
).round(1)

# -- Flag threats --------------------------------------------------------------
# WHY: anomaly_score_raw == -1 means Isolation Forest flagged this user as an
# outlier. We translate that to a human-readable THREAT / normal label.
features['is_threat'] = features['anomaly_score_raw'].apply(
    lambda x: 'THREAT' if x == -1 else 'normal'
)

# -- Evaluate against known labels ---------------------------------------------
# WHY: Even though Isolation Forest doesn't use labels during training, we CAN
# compare its output to the known labels afterward to see how well it did.
# This is called "post-hoc evaluation" and it's valid -- we're not leaking labels
# into the model, just checking its accuracy after the fact.
if 'label' in features.columns:
    threats_detected = features[
        (features['is_threat'] == 'THREAT') & (features['label'] == 1)
    ]
    total_attackers = features['label'].sum()
    print(f"\n[INFO] Post-hoc evaluation vs known labels:")
    print(f"   Known attackers  : {total_attackers}")
    print(f"   Flagged as THREAT: {(features['is_threat'] == 'THREAT').sum()}")
    print(f"   Attackers caught : {len(threats_detected)} / {total_attackers}")

# -- Save results --------------------------------------------------------------
os.makedirs('data', exist_ok=True)
features.to_csv('data/risk_scores.csv', index=False)

threats = features[features['is_threat'] == 'THREAT'].sort_values(
    'risk_score_0_100', ascending=False
)

print(f"\nDetected {len(threats)} suspicious users out of {len(features)} total")
print("\nTop 10 highest risk users:")
print(threats[[
    'user', 'risk_score_0_100', 'failed_logins', 'sensitive_files',
    'usb_events', 'bytes_sent', 'suspicious_conns'
]].head(10).to_string(index=False))
print("\n[OK] risk_scores.csv saved")


def run_anomaly_detection():
    """Called by main.py -- returns the features DataFrame with risk scores."""
    return features