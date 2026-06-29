"""
XDR Feature Extractor — Fast Vectorized Version
================================================
Reads raw logs and builds one row per user with 14 features:

SIEM  (3): failed_logins_count, login_success_after_failure, new_ip_flag
EDR   (3): sensitive_files_accessed, files_vs_role_threshold, file_access_spike
NDR   (3): bytes_sent, bytes_vs_role_threshold, dst_ip_suspicious
XDR   (5): login_hour_deviation, bytes_vs_personal_baseline,
           file_to_transfer_gap_mins, login_hour, chain_detected

Output: data/features.csv
"""

import pandas as pd
import numpy as np
import os

print("Loading logs...")
auth     = pd.read_csv('data/auth_logs.csv',     parse_dates=['timestamp'])
endpoint = pd.read_csv('data/endpoint_logs.csv', parse_dates=['timestamp'])
network  = pd.read_csv('data/network_logs.csv',  parse_dates=['timestamp'])

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SUSPICIOUS_IPS    = ["185.220.101.5", "45.33.32.156", "91.108.4.1", "203.0.113.50"]
SENSITIVE_PATTERN = 'confidential|secret|finance|salary|strategic|employee|private_key|password'

ROLE_FILE_THRESHOLD = {
    'admin': 40, 'it': 25, 'manager': 15,
    'support': 8, 'junior': 8, 'employee': 6
}
ROLE_BYTES_THRESHOLD = {
    'admin': 2_000_000_000, 'it': 1_000_000_000, 'manager': 200_000_000,
    'support': 100_000_000, 'junior': 100_000_000, 'employee': 100_000_000
}

def get_role(user):
    if user.startswith('admin'): return 'admin'
    if user.startswith('it'):    return 'it'
    if user.startswith('atk1'):
        idx = int(user.split('_')[1])
        return ['admin', 'it', 'manager'][idx % 3]
    if user.startswith('atk2'):
        idx = int(user.split('_')[1])
        return 'support' if idx % 2 == 0 else 'junior'
    return 'employee'

# ─────────────────────────────────────────────────────────────
# PRE-PROCESS — add helper columns once for all users
# ─────────────────────────────────────────────────────────────
print("Pre-processing...")

BASELINE_CUTOFF = auth['timestamp'].min() + pd.Timedelta(days=60)

auth['date']     = auth['timestamp'].dt.date
auth['hour']     = auth['timestamp'].dt.hour
auth['is_susp_ip'] = auth['src_ip'].isin(SUSPICIOUS_IPS)

endpoint['date']      = endpoint['timestamp'].dt.date
endpoint['is_sensitive'] = endpoint['file'].str.contains(
    SENSITIVE_PATTERN, case=False, na=False)

network['date']       = network['timestamp'].dt.date
network['is_susp_dst'] = network['dest_ip'].isin(SUSPICIOUS_IPS)

# Baseline versions
auth_base = auth[auth['timestamp'] < BASELINE_CUTOFF]
net_base  = network[network['timestamp'] < BASELINE_CUTOFF]
end_base  = endpoint[endpoint['timestamp'] < BASELINE_CUTOFF]

# ─────────────────────────────────────────────────────────────
# VECTORIZED AGGREGATIONS — compute all at once per user
# ─────────────────────────────────────────────────────────────
print("Computing aggregations...")

# SIEM — max daily failed logins per user
auth_failed = auth[auth['status'] == 'failed']
max_daily_fail = (auth_failed
    .groupby(['user', 'date']).size()
    .groupby('user').max()
    .rename('failed_logins_count'))

# SIEM — login success after failure
users_with_fail    = set(auth_failed['user'].unique())
users_with_success = set(auth[auth['status'] == 'success']['user'].unique())
success_after_fail = pd.Series(
    {u: 1 if u in users_with_fail and u in users_with_success else 0
     for u in auth['user'].unique()},
    name='login_success_after_failure'
)

# SIEM — median login hour (successes only)
median_login_hour = (auth[auth['status'] == 'success']
    .groupby('user')['hour'].median()
    .rename('login_hour'))

# SIEM — new IP flag (IP seen after baseline that wasn't in baseline)
baseline_ips_per_user = auth_base.groupby('user')['src_ip'].apply(set)
all_ips_per_user      = auth.groupby('user')['src_ip'].apply(set)

def has_new_ip(user):
    baseline = baseline_ips_per_user.get(user, set())
    all_ips  = all_ips_per_user.get(user, set())
    return 1 if len(all_ips - baseline) > 0 else 0

all_users = sorted(auth['user'].unique())
new_ip_flag = pd.Series(
    {u: has_new_ip(u) for u in all_users},
    name='new_ip_flag'
)

# EDR — total sensitive files
total_sensitive = (endpoint[endpoint['is_sensitive']]
    .groupby('user').size()
    .rename('sensitive_files_accessed'))

# EDR — file_access_spike (peak day vs baseline avg)
baseline_daily_sens = (end_base[end_base['is_sensitive']]
    .groupby(['user', 'date']).size()
    .groupby('user').mean()
    .rename('baseline_sens_avg'))

peak_daily_sens = (endpoint[endpoint['is_sensitive']]
    .groupby(['user', 'date']).size()
    .groupby('user').max()
    .rename('peak_sens_day'))

# NDR — total bytes sent
total_bytes = (network.groupby('user')['bytes_sent'].sum()
    .rename('bytes_sent'))

# NDR — peak daily bytes
peak_daily_bytes = (network.groupby(['user', 'date'])['bytes_sent']
    .sum().groupby('user').max()
    .rename('peak_day_bytes'))

# NDR — suspicious destination
susp_conns = (network[network['is_susp_dst']]
    .groupby('user').size()
    .rename('susp_conns'))

# XDR — login_hour_deviation
baseline_mean_hour = (auth_base[auth_base['status'] == 'success']
    .groupby('user')['hour'].mean()
    .rename('baseline_mean_hour'))

# XDR — bytes_vs_personal_baseline
baseline_bytes_avg = (net_base.groupby(['user', 'date'])['bytes_sent']
    .sum().groupby('user').mean()
    .rename('baseline_bytes_avg'))

# ─────────────────────────────────────────────────────────────
# CHAIN DETECTION — vectorized
# suspicious_login_days ∩ sensitive_file_days ∩ network_days
# ─────────────────────────────────────────────────────────────
print("Computing chain detection...")

BASELINE_CUTOFF_DATE = (auth['timestamp'].min() + pd.Timedelta(days=60)).date()

# Baseline IPs per user (vectorized)
auth['is_baseline'] = auth['timestamp'].dt.date < BASELINE_CUTOFF_DATE
baseline_ip_set = (auth[auth['is_baseline']]
    .groupby('user')['src_ip'].apply(set))

# Flag suspicious logins: new IP (not in baseline) OR very early hours
def flag_susp(row):
    b_ips = baseline_ip_set.get(row['user'], set())
    return (row['src_ip'] not in b_ips) or (row['hour'] < 6)

auth['is_susp_login'] = (
    (auth['status'] == 'success') &
    (auth['is_susp_ip'] | (auth['hour'] < 6))
)

# Get suspicious login events with timestamps
susp_logins_df = auth[auth['is_susp_login']][['user','timestamp','date']].copy()
susp_logins_df = susp_logins_df.rename(columns={'timestamp':'login_ts'})

# Get sensitive file events
sens_df = endpoint[endpoint['is_sensitive']][['user','timestamp','date']].copy()
sens_df = sens_df.rename(columns={'timestamp':'file_ts'})

# Get network transfer events
net_df = network[['user','timestamp','date']].copy()
net_df = net_df.rename(columns={'timestamp':'net_ts'})

# Merge: login → files on same date
chain1 = susp_logins_df.merge(sens_df, on=['user','date'])
chain1 = chain1[
    (chain1['file_ts'] > chain1['login_ts']) &
    (chain1['file_ts'] <= chain1['login_ts'] + pd.Timedelta(minutes=90))
]

if len(chain1) > 0:
    # For each login-file pair, find transfer within 60 mins
    chain1_max = chain1.groupby(['user','date','login_ts'])['file_ts'].max().reset_index()
    chain1_max = chain1_max.rename(columns={'file_ts':'last_file_ts'})
    chain2 = chain1_max.merge(net_df, on=['user','date'])
    chain2 = chain2[
        (chain2['net_ts'] > chain2['last_file_ts']) &
        (chain2['net_ts'] <= chain2['last_file_ts'] + pd.Timedelta(minutes=60))
    ]
    users_with_chain = set(chain2['user'].unique())
else:
    users_with_chain = set()

chain_detected = pd.Series(
    {u: 1 if u in users_with_chain else 0 for u in all_users},
    name='chain_detected'
)

# XDR — file_to_transfer_gap (min gap between sensitive file and transfer on same day)
print("Computing file-to-transfer gaps...")

sens_last_per_day = (endpoint[endpoint['is_sensitive']]
    .groupby(['user', 'date'])['timestamp'].max()
    .reset_index()
    .rename(columns={'timestamp': 'last_file_time'}))

net_first_per_day = (network
    .groupby(['user', 'date'])['timestamp'].min()
    .reset_index()
    .rename(columns={'timestamp': 'first_net_time'}))

gap_df = sens_last_per_day.merge(net_first_per_day, on=['user', 'date'], how='inner')
gap_df['gap_mins'] = (
    gap_df['first_net_time'] - gap_df['last_file_time']
).dt.total_seconds() / 60
gap_df = gap_df[gap_df['gap_mins'] > 0]

min_gap_per_user = (gap_df.groupby('user')['gap_mins']
    .min().round(0).astype(int)
    .rename('file_to_transfer_gap_mins'))

# ─────────────────────────────────────────────────────────────
# BUILD FINAL FEATURE TABLE
# ─────────────────────────────────────────────────────────────
print("Building feature table...")

features = pd.DataFrame({'user': all_users}).set_index('user')

features = features.join(max_daily_fail)
features = features.join(success_after_fail)
features = features.join(median_login_hour)
features = features.join(new_ip_flag)
features = features.join(total_sensitive)
features = features.join(peak_daily_sens)
features = features.join(baseline_daily_sens)
features = features.join(total_bytes)
features = features.join(peak_daily_bytes)
features = features.join(susp_conns)
features = features.join(baseline_mean_hour)
features = features.join(baseline_bytes_avg)
features = features.join(min_gap_per_user)
features = features.join(chain_detected)

features = features.fillna(0)
features = features.reset_index()

# ─────────────────────────────────────────────────────────────
# COMPUTE RATIO FEATURES
# ─────────────────────────────────────────────────────────────
features['role'] = features['user'].apply(get_role)

features['files_vs_role_threshold'] = features.apply(
    lambda r: round(r['peak_sens_day'] / ROLE_FILE_THRESHOLD.get(r['role'], 6), 3), axis=1)

features['bytes_vs_role_threshold'] = features.apply(
    lambda r: round(r['peak_day_bytes'] / ROLE_BYTES_THRESHOLD.get(r['role'], 100_000_000), 3), axis=1)

features['file_access_spike'] = features.apply(
    lambda r: round(r['peak_sens_day'] / r['baseline_sens_avg'], 3)
    if r['baseline_sens_avg'] > 0 else float(r['peak_sens_day']), axis=1)

features['login_hour_deviation'] = features.apply(
    lambda r: round(abs(r['login_hour'] - r['baseline_mean_hour']), 3)
    if r['baseline_mean_hour'] > 0 else 0.0, axis=1)

features['bytes_vs_personal_baseline'] = features.apply(
    lambda r: round(r['peak_day_bytes'] / r['baseline_bytes_avg'], 3)
    if r['baseline_bytes_avg'] > 0 else 0.0, axis=1)

features['dst_ip_suspicious'] = (features['susp_conns'] > 0).astype(int)

features['file_to_transfer_gap_mins'] = features['file_to_transfer_gap_mins'].replace(0, 999)

# ─────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────
def assign_label(user):
    if user.startswith('atk1'): return 1
    if user.startswith('atk2'): return 2
    return 0

features['label'] = features['user'].apply(assign_label)

# ─────────────────────────────────────────────────────────────
# SELECT FINAL COLUMNS
# ─────────────────────────────────────────────────────────────
final_cols = [
    'user',
    'failed_logins_count', 'login_success_after_failure',
    'login_hour', 'new_ip_flag',
    'sensitive_files_accessed', 'files_vs_role_threshold', 'file_access_spike',
    'bytes_sent', 'bytes_vs_role_threshold', 'dst_ip_suspicious',
    'login_hour_deviation', 'bytes_vs_personal_baseline',
    'file_to_transfer_gap_mins', 'chain_detected',
    'label'
]
features = features[final_cols]

os.makedirs('data', exist_ok=True)
features.to_csv('data/features.csv', index=False)

print(f"\n[OK] features.csv -> {features.shape[0]} users x {features.shape[1]} columns")
print(f"   Normal  : {(features['label']==0).sum()}")
print(f"   Attack1 : {(features['label']==1).sum()}")
print(f"   Attack2 : {(features['label']==2).sum()}")
print(f"\nchain_detected counts:")
print(f"   Normal  : {features[features['label']==0]['chain_detected'].sum()} / {(features['label']==0).sum()}")
print(f"   Attack1 : {features[features['label']==1]['chain_detected'].sum()} / {(features['label']==1).sum()}")
print(f"   Attack2 : {features[features['label']==2]['chain_detected'].sum()} / {(features['label']==2).sum()}")


def extract_features():
    return features