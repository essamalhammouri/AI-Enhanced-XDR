"""
XDR Dataset Generator — Maximum Difficulty Version
===================================================
Heavy overlap on ALL features. No single feature solves the problem.
XDR must combine multiple signals. Baseline will be harder.

This is the raw dataset — you push performance up with model tuning.
"""

import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)


def rng(mean, std, low, high, size=1, rs=None):
    if rs is None:
        rs = np.random
    vals = rs.normal(mean, std, size * 10)
    vals = vals[(vals >= low) & (vals <= high)]
    if len(vals) < size:
        vals = np.clip(rs.normal(mean, std, size), low, high)
    return vals[:size]


def choice_weighted(options, weights, size=1, rs=None):
    if rs is None:
        rs = np.random
    return rs.choice(options, size=size, p=weights)


ROLES = {
    'admin':    {'file_thresh': 40, 'bytes_thresh': 2_000_000_000},
    'it':       {'file_thresh': 25, 'bytes_thresh': 1_000_000_000},
    'manager':  {'file_thresh': 15, 'bytes_thresh':   200_000_000},
    'support':  {'file_thresh': 8,  'bytes_thresh':   100_000_000},
    'employee': {'file_thresh': 6,  'bytes_thresh':   100_000_000},
}


def generate_normal(n, rs):
    records = []
    for i in range(n):
        role = choice_weighted(
            ['employee', 'support', 'it', 'manager', 'admin'],
            [0.38, 0.18, 0.20, 0.12, 0.12], rs=rs
        )[0]
        ft = ROLES[role]['file_thresh']
        bt = ROLES[role]['bytes_thresh']

        user_type = choice_weighted(
            ['standard', 'busy', 'traveler', 'late_worker', 'noisy_auth',
             'busy_admin', 'suspicious_normal'],
            [0.22, 0.17, 0.11, 0.12, 0.08, 0.18, 0.12], rs=rs
        )[0]

        if user_type in ['busy_admin', 'suspicious_normal'] and role not in ['admin', 'it']:
            user_type = 'busy'

        # ── SIEM ──────────────────────────────────────────────
        if user_type == 'noisy_auth':
            failed = int(rng(11, 4, 4, 22, rs=rs)[0])
            new_ip = int(rs.choice([0, 1], p=[0.70, 0.30]))
            login_hour = int(rng(9, 2.5, 4, 20, rs=rs)[0])
        elif user_type == 'traveler':
            failed = int(rng(4, 2, 0, 9, rs=rs)[0])
            new_ip = int(rs.choice([0, 1], p=[0.20, 0.80]))
            login_hour = int(rng(10, 4, 3, 22, rs=rs)[0])
        elif user_type == 'late_worker':
            failed = int(rng(5, 2, 0, 12, rs=rs)[0])
            new_ip = int(rs.choice([0, 1], p=[0.50, 0.50]))
            login_hour = int(rng(20, 3, 15, 23, rs=rs)[0])
        else:
            failed = int(rng(3.5, 2.5, 0, 12, rs=rs)[0])
            new_ip = int(rs.choice([0, 1], p=[0.70, 0.30]))
            login_hour = int(rng(11, 3.5, 5, 22, rs=rs)[0])

        # ── EDR — heavy overlap ───────────────────────────────
        if user_type == 'suspicious_normal':
            file_spike = round(rng(3.0, 1.20, 1.0, 6.0, rs=rs)[0], 3)
            admin_tools = int(rng(5.0, 2.0, 2, 10, rs=rs)[0])
            sens_files_raw = int(rng(ft * 0.75, ft * 0.15, ft * 0.45, ft * 0.98, rs=rs)[0])
        elif user_type == 'busy_admin':
            file_spike = round(rng(2.5, 1.10, 0.8, 5.5, rs=rs)[0], 3)
            admin_tools = int(rng(4.0, 2.0, 1, 9, rs=rs)[0])
            sens_files_raw = int(rng(ft * 0.65, ft * 0.20, ft * 0.30, ft * 0.95, rs=rs)[0])
        elif user_type == 'busy' or role in ['admin', 'it']:
            file_spike = round(rng(1.8, 0.80, 0.3, 4.5, rs=rs)[0], 3)
            if role in ['admin', 'it']:
                admin_tools = int(rng(3.5, 2.0, 0, 8, rs=rs)[0])
            else:
                admin_tools = int(rng(1.5, 1.2, 0, 5, rs=rs)[0])
            sens_files_raw = int(rng(ft * 0.55, ft * 0.25, 1, ft * 0.92, rs=rs)[0])
        else:
            file_spike = round(rng(1.2, 0.70, 0.1, 3.8, rs=rs)[0], 3)
            admin_tools = int(rng(1.0, 1.2, 0, 4, rs=rs)[0])
            sens_files_raw = int(rng(ft * 0.35, ft * 0.22, 0, ft * 0.80, rs=rs)[0])

        files_vs_role = round(sens_files_raw / ft, 3) if ft > 0 else 0

        # ── NDR — heavy overlap ───────────────────────────────
        if user_type == 'suspicious_normal':
            bytes_raw = int(rng(bt * 0.72, bt * 0.18, bt * 0.40, bt * 0.95, rs=rs)[0])
            distinct_hosts = int(rng(5, 1.8, 2, 9, rs=rs)[0])
        elif user_type == 'busy_admin':
            bytes_raw = int(rng(bt * 0.60, bt * 0.22, bt * 0.25, bt * 0.95, rs=rs)[0])
            distinct_hosts = int(rng(4, 1.8, 2, 8, rs=rs)[0])
        elif role in ['admin', 'it'] or user_type == 'busy':
            bytes_raw = int(rng(bt * 0.50, bt * 0.28, bt * 0.05, bt * 1.10, rs=rs)[0])
            distinct_hosts = int(rng(3, 1.5, 1, 7, rs=rs)[0])
        else:
            bytes_raw = int(rng(bt * 0.35, bt * 0.25, bt * 0.02, bt * 0.85, rs=rs)[0])
            distinct_hosts = int(rng(2, 1.2, 1, 5, rs=rs)[0])

        bytes_vs_role = round(bytes_raw / bt, 3)

        if role in ['admin', 'it']:
            dst_cat = int(choice_weighted([0, 1, 2], [0.45, 0.40, 0.15], rs=rs)[0])
        else:
            dst_cat = int(choice_weighted([0, 1, 2], [0.55, 0.35, 0.10], rs=rs)[0])

        # ── XDR-only — HEAVY overlap ──────────────────────────
        if user_type == 'traveler':
            hour_dev = round(rng(2.5, 1.5, 0.3, 6.5, rs=rs)[0], 3)
        elif user_type == 'late_worker':
            hour_dev = round(rng(1.5, 1.2, 0.0, 5.0, rs=rs)[0], 3)
        elif user_type == 'suspicious_normal':
            hour_dev = round(rng(1.8, 1.3, 0.0, 5.5, rs=rs)[0], 3)
        else:
            hour_dev = round(rng(1.0, 0.9, 0.0, 4.0, rs=rs)[0], 3)

        # bytes_vs_personal — up to 3.5 for normals (overlaps attacks)
        if user_type == 'suspicious_normal':
            bytes_personal = round(rng(2.6, 0.80, 1.3, 4.0, rs=rs)[0], 3)
        elif user_type == 'busy_admin':
            bytes_personal = round(rng(2.0, 0.70, 0.80, 3.5, rs=rs)[0], 3)
        elif user_type == 'busy':
            bytes_personal = round(rng(1.6, 0.60, 0.55, 3.0, rs=rs)[0], 3)
        else:
            bytes_personal = round(rng(1.2, 0.55, 0.35, 2.80, rs=rs)[0], 3)

        # gap — heavy overlap with attacks
        if user_type == 'suspicious_normal':
            gap = int(rng(80, 55, 20, 280, rs=rs)[0])
        elif user_type in ['busy_admin', 'busy']:
            gap = int(rng(130, 80, 25, 400, rs=rs)[0])
        else:
            gap = int(rng(200, 110, 30, 500, rs=rs)[0])

        # sensitive_file_category_deviation — normals up to 0.80
        if user_type == 'suspicious_normal':
            cat_dev = round(rng(0.58, 0.22, 0.20, 0.95, rs=rs)[0], 3)
        elif user_type == 'busy_admin':
            cat_dev = round(rng(0.40, 0.22, 0.05, 0.80, rs=rs)[0], 3)
        elif user_type == 'busy':
            cat_dev = round(rng(0.30, 0.20, 0.0, 0.70, rs=rs)[0], 3)
        else:
            cat_dev = round(rng(0.20, 0.18, 0.0, 0.65, rs=rs)[0], 3)

        # role_based_access_score — normals up to 0.70
        if user_type == 'suspicious_normal':
            role_score = round(rng(0.45, 0.18, 0.15, 0.80, rs=rs)[0], 3)
        elif user_type == 'busy_admin':
            role_score = round(rng(0.32, 0.17, 0.08, 0.68, rs=rs)[0], 3)
        elif user_type == 'busy':
            role_score = round(rng(0.25, 0.15, 0.05, 0.60, rs=rs)[0], 3)
        else:
            role_score = round(rng(0.18, 0.13, 0.02, 0.55, rs=rs)[0], 3)

        records.append({
            'failed_logins_count': failed, 'new_ip_flag': new_ip, 'login_hour': login_hour,
            'files_vs_role_threshold': files_vs_role, 'file_access_spike': file_spike,
            'admin_tools_used_count': admin_tools,
            'bytes_vs_role_threshold': bytes_vs_role, 'distinct_hosts_accessed': distinct_hosts,
            'dst_ip_category': dst_cat,
            'login_hour_deviation': hour_dev, 'bytes_vs_personal_baseline': bytes_personal,
            'file_to_transfer_gap_mins': gap,
            'sensitive_file_category_deviation': cat_dev,
            'role_based_access_score': role_score,
            'attack_subtype': 'normal_' + user_type, 'label': 0
        })
    return pd.DataFrame(records)


def generate_attack1(n, rs, allowed_methods=None):
    if allowed_methods is None:
        allowed_methods = ['phishing', 'spray', 'credential_stuffing']
    records = []
    for i in range(n):
        role = choice_weighted(['admin', 'it', 'manager'], [0.35, 0.40, 0.25], rs=rs)[0]
        ft = ROLES[role]['file_thresh']
        bt = ROLES[role]['bytes_thresh']

        method = rs.choice(allowed_methods)
        stealthy = rs.choice([False, True], p=[0.80, 0.20])  # more stealthy

        # ── SIEM ─────────────────────────────────────────────
        if method == 'phishing':
            failed = int(rng(1.5, 1.5, 0, 6, rs=rs)[0])
            login_hour = int(rng(rs.choice([3, 10, 14]), 3.0, 0, 22, rs=rs)[0])
        elif method == 'spray':
            failed = int(rng(4, 2.5, 0, 11, rs=rs)[0])
            login_hour = int(rng(rs.choice([4, 9, 15]), 3.0, 0, 22, rs=rs)[0])
        else:
            failed = int(rng(6, 3, 1, 14, rs=rs)[0])
            login_hour = int(rng(rs.choice([2, 11, 16]), 3.0, 0, 22, rs=rs)[0])

        # new_ip now 60-70% not 100% — residential proxies exist
        new_ip = int(rs.choice([0, 1], p=[0.30, 0.70])) if not stealthy \
                 else int(rs.choice([0, 1], p=[0.55, 0.45]))

        # ── EDR — heavy overlap ──────────────────────────────
        personal_baseline = int(rng(ft * 0.32, ft * 0.12, ft * 0.10, ft * 0.55, rs=rs)[0])
        sens_files_raw = int(rng(
            personal_baseline * 1.8, personal_baseline * 0.50,
            personal_baseline + 1, int(ft * 0.72), rs=rs
        )[0])
        sens_files_raw = max(sens_files_raw, personal_baseline + 1)
        files_vs_role = round(sens_files_raw / ft, 3)
        file_spike = round(rng(1.7, 0.80, 0.6, 4.5, rs=rs)[0], 3)
        admin_tools = int(rng(1.5, 1.5, 0, 5, rs=rs)[0])

        # ── NDR ──────────────────────────────────────────────
        personal_bytes = int(rng(bt * 0.40, bt * 0.18, bt * 0.08, bt * 0.72, rs=rs)[0])
        bytes_raw = int(rng(
            personal_bytes * 1.7, personal_bytes * 0.45,
            personal_bytes + 40_000, int(bt * 0.80), rs=rs
        )[0])
        bytes_vs_role = round(bytes_raw / bt, 3)
        distinct_hosts = int(rng(2.5, 1.2, 1, 5, rs=rs)[0])
        dst_cat = int(choice_weighted([0, 1, 2], [0.30, 0.40, 0.30], rs=rs)[0])

        # ── XDR-only — OVERLAPS with normals ─────────────────
        if stealthy:
            hour_dev = round(rng(1.3, 0.9, 0.2, 4.0, rs=rs)[0], 3)
            bytes_personal = round(rng(1.7, 0.55, 1.0, 3.2, rs=rs)[0], 3)
            gap = int(rng(130, 60, 40, 260, rs=rs)[0])
            cat_dev = round(rng(0.45, 0.22, 0.15, 0.85, rs=rs)[0], 3)
            role_score = round(rng(0.50, 0.20, 0.20, 0.85, rs=rs)[0], 3)
        else:
            hour_dev = round(rng(2.8, 1.6, 0.4, 7.5, rs=rs)[0], 3)
            bytes_personal = round(rng(2.2, 0.70, 1.1, 4.0, rs=rs)[0], 3)
            gap = int(rng(60, 35, 10, 180, rs=rs)[0])
            cat_dev = round(rng(0.58, 0.22, 0.18, 0.95, rs=rs)[0], 3)
            role_score = round(rng(0.62, 0.18, 0.30, 0.92, rs=rs)[0], 3)

        records.append({
            'failed_logins_count': failed, 'new_ip_flag': new_ip, 'login_hour': login_hour,
            'files_vs_role_threshold': files_vs_role, 'file_access_spike': file_spike,
            'admin_tools_used_count': admin_tools,
            'bytes_vs_role_threshold': bytes_vs_role, 'distinct_hosts_accessed': distinct_hosts,
            'dst_ip_category': dst_cat,
            'login_hour_deviation': hour_dev, 'bytes_vs_personal_baseline': bytes_personal,
            'file_to_transfer_gap_mins': gap,
            'sensitive_file_category_deviation': cat_dev,
            'role_based_access_score': role_score,
            'attack_subtype': 'attack1_' + method + ('_stealthy' if stealthy else ''),
            'label': 1
        })
    return pd.DataFrame(records)


def generate_attack2(n, rs, allowed_types=None):
    if allowed_types is None:
        allowed_types = ['planned_exfil', 'opportunistic', 'gradual_buildup']
    records = []
    for i in range(n):
        role = choice_weighted(['admin', 'it'], [0.65, 0.35], rs=rs)[0]
        ft = ROLES[role]['file_thresh']
        bt = ROLES[role]['bytes_thresh']

        insider_type = rs.choice(allowed_types)
        stealthy = rs.choice([False, True], p=[0.78, 0.22])  # more stealthy

        # ── SIEM ─────────────────────────────────────────────
        failed = int(rng(2.5, 1.8, 0, 9, rs=rs)[0])
        new_ip = int(rs.choice([0, 1], p=[0.85, 0.15]))
        login_hour = int(rng(rs.choice([9, 11, 14, 16]), 2.5, 6, 20, rs=rs)[0])

        # ── EDR — heavy overlap with busy_admin ──────────────
        sens_files_raw = int(rng(ft * 0.68, ft * 0.15, ft * 0.40, ft * 0.92, rs=rs)[0])
        files_vs_role = round(sens_files_raw / ft, 3)

        if insider_type == 'gradual_buildup':
            file_spike = round(rng(2.3, 0.90, 1.1, 4.8, rs=rs)[0], 3)
        else:
            file_spike = round(rng(2.8, 1.00, 1.4, 5.5, rs=rs)[0], 3)

        admin_tools = int(rng(4.0, 1.8, 1, 8, rs=rs)[0])

        # ── NDR ──────────────────────────────────────────────
        bytes_raw = int(rng(bt * 0.68, bt * 0.15, bt * 0.45, bt * 0.92, rs=rs)[0])
        bytes_vs_role = round(bytes_raw / bt, 3)

        if insider_type == 'planned_exfil':
            distinct_hosts = int(rng(5, 1.5, 2, 8, rs=rs)[0])
        else:
            distinct_hosts = int(rng(4, 1.5, 2, 7, rs=rs)[0])

        dst_cat = int(choice_weighted([0, 1, 2], [0.35, 0.45, 0.20], rs=rs)[0])

        # ── XDR-only — OVERLAPS ──────────────────────────────
        hour_dev = round(rng(1.2, 0.8, 0.0, 3.5, rs=rs)[0], 3)

        if stealthy:
            bytes_personal = round(rng(2.2, 0.55, 1.4, 3.4, rs=rs)[0], 3)
            gap = int(rng(140, 70, 35, 280, rs=rs)[0])
            cat_dev = round(rng(0.55, 0.20, 0.25, 0.85, rs=rs)[0], 3)
            role_score = round(rng(0.52, 0.20, 0.20, 0.85, rs=rs)[0], 3)
        elif insider_type == 'gradual_buildup':
            bytes_personal = round(rng(2.6, 0.65, 1.6, 4.2, rs=rs)[0], 3)
            gap = int(rng(80, 40, 25, 180, rs=rs)[0])
            cat_dev = round(rng(0.65, 0.18, 0.35, 0.95, rs=rs)[0], 3)
            role_score = round(rng(0.63, 0.17, 0.30, 0.92, rs=rs)[0], 3)
        else:
            bytes_personal = round(rng(3.0, 0.80, 1.8, 4.8, rs=rs)[0], 3)
            gap = int(rng(45, 25, 10, 110, rs=rs)[0])
            cat_dev = round(rng(0.70, 0.16, 0.40, 0.98, rs=rs)[0], 3)
            role_score = round(rng(0.72, 0.15, 0.40, 0.98, rs=rs)[0], 3)

        records.append({
            'failed_logins_count': failed, 'new_ip_flag': new_ip, 'login_hour': login_hour,
            'files_vs_role_threshold': files_vs_role, 'file_access_spike': file_spike,
            'admin_tools_used_count': admin_tools,
            'bytes_vs_role_threshold': bytes_vs_role, 'distinct_hosts_accessed': distinct_hosts,
            'dst_ip_category': dst_cat,
            'login_hour_deviation': hour_dev, 'bytes_vs_personal_baseline': bytes_personal,
            'file_to_transfer_gap_mins': gap,
            'sensitive_file_category_deviation': cat_dev,
            'role_based_access_score': role_score,
            'attack_subtype': 'attack2_' + insider_type + ('_stealthy' if stealthy else ''),
            'label': 2
        })
    return pd.DataFrame(records)


def generate_dataset(n_normal=500, n_attack1=250, n_attack2=250,
                     seed=None, a1_methods=None, a2_types=None):
    rs = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    df_n  = generate_normal(n_normal, rs)
    df_a1 = generate_attack1(n_attack1, rs, allowed_methods=a1_methods)
    df_a2 = generate_attack2(n_attack2, rs, allowed_types=a2_types)
    df = pd.concat([df_n, df_a1, df_a2], ignore_index=True)
    df = df.sample(frac=1, random_state=rs.randint(0, 1_000_000)).reset_index(drop=True)
    return df


if __name__ == '__main__':
    print("Generating XDR dataset — maximum difficulty version...")
    df = generate_dataset()
    print(f"\nDataset: {len(df)} samples")
    print(f"  Normal  : {(df['label']==0).sum()}")
    print(f"  Attack1 : {(df['label']==1).sum()}")
    print(f"  Attack2 : {(df['label']==2).sum()}")
    df.to_csv('data/features.csv', index=False)
    print("\n✅ features.csv saved")