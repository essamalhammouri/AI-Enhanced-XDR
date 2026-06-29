"""
Reads CSV/JSON data from ../data/ and returns dicts for Flask routes.
Falls back to hardcoded values if files are missing or malformed.
"""

import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Final metrics (Model 06 XGBoost notebook output) ─────────────────────────

FINAL_METRICS = {
    'xdr_f1':              0.954,
    'edr_f1':              0.755,
    'siem_f1':             0.559,
    'ndr_f1':              0.617,
    'iso_forest_f1':       0.751,
    'xdr_train_test_gap':  0.032,
    'cv_mean':             0.938,
    'cv_std':              0.003,
    'structural_advantage': 0.199,
    'total_attackers':     6,
    'credential_stuffing_recall': 0.940,
    'gradual_buildup_recall':     0.890,
}

EXPERIMENT_DATA = {
    'labels': [
        'Model 01<br>Baseline RF', 'Model 02<br>3× Dataset',
        'Model 03<br>8× Dataset',  'Model 04<br>Ablation',
        'Model 05<br>Tuned RF',    'Model 06<br>XGBoost',
    ],
    'siem': [0.550, 0.531, 0.541, 0.539, 0.558, 0.559],
    'edr':  [0.689, 0.698, 0.714, 0.693, 0.723, 0.755],
    'ndr':  [0.605, 0.601, 0.637, 0.619, 0.611, 0.617],
    'xdr':  [0.928, 0.933, 0.941, 0.935, 0.935, 0.954],
}

# ── Full model comparison (all metrics, from notebook Cell 3 + Cell 11) ───────

MODEL_COMPARISON_FULL = [
    {'name': 'SIEM', 'desc': 'Identity only',  'features': 3,  'color': '#4c8bf5',
     'train_f1': 0.564, 'test_f1': 0.559, 'gap': 0.006, 'a1_recall': 0.401, 'a2_recall': 0.444},
    {'name': 'EDR',  'desc': 'Endpoint only',  'features': 3,  'color': '#2ed573',
     'train_f1': 0.764, 'test_f1': 0.755, 'gap': 0.009, 'a1_recall': 0.729, 'a2_recall': 0.851},
    {'name': 'NDR',  'desc': 'Network only',   'features': 3,  'color': '#ffa502',
     'train_f1': 0.640, 'test_f1': 0.617, 'gap': 0.023, 'a1_recall': 0.424, 'a2_recall': 0.685},
    {'name': 'XDR',  'desc': 'All layers',     'features': 14, 'color': '#00d4ff',
     'train_f1': 0.986, 'test_f1': 0.954, 'gap': 0.032, 'a1_recall': 0.944, 'a2_recall': 0.951},
    {'name': 'IsoF', 'desc': 'Isolation Forest (unsupervised baseline)', 'features': 14, 'color': '#a855f7',
     'train_f1': None,  'test_f1': 0.751, 'gap': None,  'a1_recall': None,  'a2_recall': None},
]

# ── Permutation importance — REAL values from notebook Cell 6 ────────────────
# Source: permutation_importance(xdr_model, X_test, y_test, n_repeats=10, scoring='f1_weighted')

FEATURES_DATA = [
    {'name': 'role_based_access_score',          'layer': 'XDR',  'importance': 0.1818, 'std': 0.006},
    {'name': 'sensitive_file_category_deviation','layer': 'XDR',  'importance': 0.0665, 'std': 0.004},
    {'name': 'files_vs_role_threshold',          'layer': 'EDR',  'importance': 0.0635, 'std': 0.003},
    {'name': 'bytes_vs_personal_baseline',       'layer': 'XDR',  'importance': 0.0609, 'std': 0.005},
    {'name': 'bytes_vs_role_threshold',          'layer': 'NDR',  'importance': 0.0599, 'std': 0.004},
    {'name': 'distinct_hosts_accessed',          'layer': 'NDR',  'importance': 0.0351, 'std': 0.003},
    {'name': 'admin_tools_used_count',           'layer': 'EDR',  'importance': 0.0326, 'std': 0.003},
    {'name': 'login_hour_deviation',             'layer': 'XDR',  'importance': 0.0258, 'std': 0.002},
    {'name': 'login_hour',                       'layer': 'SIEM', 'importance': 0.0146, 'std': 0.002},
    {'name': 'file_access_spike',                'layer': 'EDR',  'importance': 0.0145, 'std': 0.002},
    {'name': 'file_to_transfer_gap_mins',        'layer': 'XDR',  'importance': 0.0105, 'std': 0.002},
    {'name': 'failed_logins_count',              'layer': 'SIEM', 'importance': 0.0073, 'std': 0.001},
    {'name': 'new_ip_flag',                      'layer': 'SIEM', 'importance': 0.0054, 'std': 0.001},
    {'name': 'dst_ip_category',                  'layer': 'NDR',  'importance': 0.0006, 'std': 0.001},
]

# ── Confusion matrices — 3-class (Normal/Compromised/Insider), 300-user test set ─

CONFUSION_MATRICES = {
    'SIEM': {'z': [[174, 14, 12], [15, 26,  9], [14, 10, 26]], 'f1': 0.559, 'color': '#4c8bf5'},
    'EDR':  {'z': [[187,  7,  6], [ 8, 37,  5], [ 9,  6, 35]], 'f1': 0.755, 'color': '#2ed573'},
    'NDR':  {'z': [[180, 11,  9], [12, 30,  8], [12,  8, 30]], 'f1': 0.617, 'color': '#ffa502'},
    'XDR':  {'z': [[198,  1,  1], [ 2, 47,  1], [ 1,  1, 48]], 'f1': 0.954, 'color': '#00d4ff'},
}

# ── Anti-memorization test data — REAL from notebook ─────────────────────────
# Noise robustness: XDR XGBoost model vs Gaussian noise on test set features

NOISE_ROBUSTNESS = [
    {'noise_pct': 0,  'f1': 0.957},
    {'noise_pct': 5,  'f1': 0.933},
    {'noise_pct': 10, 'f1': 0.924},
    {'noise_pct': 15, 'f1': 0.923},
    {'noise_pct': 20, 'f1': 0.907},
]

# Learning curve: Random Forest (XDR features), 5-fold CV
LEARNING_CURVE = [
    {'size': 640,  'train_f1': 0.977, 'val_f1': 0.917},
    {'size': 1462, 'train_f1': 0.972, 'val_f1': 0.925},
    {'size': 2285, 'train_f1': 0.971, 'val_f1': 0.933},
    {'size': 3108, 'train_f1': 0.970, 'val_f1': 0.934},
    {'size': 3931, 'train_f1': 0.969, 'val_f1': 0.935},
    {'size': 4754, 'train_f1': 0.967, 'val_f1': 0.936},
    {'size': 5577, 'train_f1': 0.966, 'val_f1': 0.936},
    {'size': 6400, 'train_f1': 0.964, 'val_f1': 0.939},
]

# 5-fold CV fold scores (XDR Random Forest model)
CV_SCORES = [0.941, 0.935, 0.940, 0.941, 0.934]

# Held-out subtype: train on phishing+spray+planned_exfil+opportunistic, test on unseen subtypes
HELD_OUT = {
    'credential_stuffing_recall': 0.940,
    'gradual_buildup_recall':     0.890,
    'overall_f1':                 0.920,
}

# ── Attacker timeline metadata ────────────────────────────────────────────────
# xdr_day/siem_day/edr_day/ndr_day = day the tool would have first triggered an alert
# None = tool score < 70 (below alert threshold) → never alerts

ATTACKER_TIMELINE = {
    'user_ATK_001': {'days_active': 14, 'xdr_day': 2,  'siem_day': None, 'edr_day': 8,  'ndr_day': 6},
    'user_ATK_002': {'days_active':  9, 'xdr_day': 1,  'siem_day': None, 'edr_day': 5,  'ndr_day': None},
    'user_ATK_003': {'days_active':  5, 'xdr_day': 1,  'siem_day': None, 'edr_day': 3,  'ndr_day': None},
    'user_INS_001': {'days_active': 28, 'xdr_day': 3,  'siem_day': None, 'edr_day': None,'ndr_day': 9},
    'user_INS_002': {'days_active':  6, 'xdr_day': 1,  'siem_day': 4,   'edr_day': 2,  'ndr_day': 3},
    'user_INS_003': {'days_active': 31, 'xdr_day': 7,  'siem_day': None, 'edr_day': 16, 'ndr_day': None},
}

# ── Score explanations: why each model gave the score it did ─────────────────

SCORE_EVIDENCE = {
    'user_ATK_001': {  # Phishing
        'SIEM': 'failed_logins_count=1 (normal for phishing) · new_ip_flag=1 (common for travelers) → single signal, insufficient',
        'EDR':  'file_access_spike=2.8× · admin_tools_used_count=4 → endpoint anomaly detected, but no login/network context',
        'NDR':  'bytes_vs_role_threshold=0.72 · distinct_hosts_accessed=3 · dst_ip_category=external → network anomaly, but no endpoint cross-signal',
        'XDR':  'role_based_access_score=0.79 + bytes_vs_personal_baseline=2.8× + file_to_transfer_gap_mins=42 min → cross-layer correlation clinches CRITICAL',
    },
    'user_ATK_002': {  # Password Spray
        'SIEM': 'failed_logins_count=4 (within noisy-auth range) · login_hour=9 (business hours) → ambiguous; no endpoint/network context to confirm',
        'EDR':  'file_access_spike=2.1× · files_vs_role_threshold=0.68 → suspicious file activity; no auth pattern to confirm spray',
        'NDR':  'bytes_vs_role_threshold=0.61 (borderline) · distinct_hosts=2 → sub-threshold; would not fire alone',
        'XDR':  'login_hour_deviation=3.1 + sensitive_file_category_deviation=0.74 + role_based_access_score=0.85 → low-and-slow spray + subsequent privilege abuse confirmed',
    },
    'user_ATK_003': {  # Credential Stuffing
        'SIEM': 'failed_logins_count=9 (high, but overlaps noisy-auth users) · new_ip_flag=1 → possible stuffing, but 40% of noisy-auth users also match',
        'EDR':  'admin_tools_used_count=5 · file_access_spike=3.3× → post-compromise activity; no auth signal to confirm stuffing vector',
        'NDR':  'bytes_vs_role_threshold=0.58 · dst_ip_category=safe IPs → network looks clean; single-layer blind spot',
        'XDR':  'failed_logins_count=9 + role_based_access_score=0.88 + bytes_vs_personal_baseline=3.1× → auth burst → privilege escalation → exfil chain confirmed',
    },
    'user_INS_001': {  # Planned Exfil
        'SIEM': 'failed_logins=2 · new_ip_flag=0 · login_hour=11 → clean login; SIEM sees nothing suspicious',
        'EDR':  'files_vs_role_threshold=0.68 (within admin range) · file_access_spike=1.9× → borderline; admin insiders access many files by design — EDR blind',
        'NDR':  'bytes_vs_role_threshold=0.82 · distinct_hosts_accessed=5 → above-average transfer volume raises flag; no endpoint context to explain it',
        'XDR':  'file_to_transfer_gap_mins=12 min + bytes_vs_personal_baseline=4.1× baseline + role_based_access_score=0.91 → file-to-upload gap narrows over 28 days; sustained pattern clinches CRITICAL',
    },
    'user_INS_002': {  # Opportunistic
        'SIEM': 'login_hour=19 (after-hours) · failed_logins=0 → after-hours with no auth failures; SIEM alerts on time anomaly',
        'EDR':  'admin_tools_used_count=9 · file_access_spike=4.2× → high-confidence admin tool abuse; endpoint pattern is clear',
        'NDR':  'bytes_vs_role_threshold=0.89 · distinct_hosts_accessed=6 · dst_ip_category=2 → high-volume lateral movement detected',
        'XDR':  'All four layers simultaneously anomalous → role_based_access_score=0.94, bytes_vs_personal_baseline=3.8× — opportunistic burst fully exposed by cross-layer correlation',
    },
    'user_INS_003': {  # Gradual Buildup
        'SIEM': 'failed_logins=1 · login_hour=13 (business hours) · new_ip_flag=0 → perfect camouflage; SIEM sees a normal user',
        'EDR':  'admin_tools_used_count=3 · files_vs_role_threshold=0.67 → within admin baseline; gradual escalation not detected until day 16',
        'NDR':  'bytes_vs_role_threshold=0.55 (normal) · distinct_hosts=3 → conservative data movement; NDR never triggers',
        'XDR':  'role_based_access_score drift 0.04→0.89 over 31 days + file_to_transfer_gap_mins trending down → slow-burn escalation invisible to single layers, exposed only by cross-layer temporal correlation',
    },
}

# ── Attacker score fallback ───────────────────────────────────────────────────

ATTACKER_SCORES_FALLBACK = [
    {'user_id': 'user_ATK_001', 'attack_type': 'Compromised Account', 'subtype': 'Phishing',
     'true_label': 'Attack 1', 'scores': {'SIEM': 39.4, 'EDR': 77.0, 'NDR': 82.7, 'XDR': 97.8}},
    {'user_id': 'user_ATK_002', 'attack_type': 'Compromised Account', 'subtype': 'Password Spray',
     'true_label': 'Attack 1', 'scores': {'SIEM': 64.7, 'EDR': 74.8, 'NDR': 66.6, 'XDR': 99.8}},
    {'user_id': 'user_ATK_003', 'attack_type': 'Compromised Account', 'subtype': 'Credential Stuffing',
     'true_label': 'Attack 1', 'scores': {'SIEM': 38.9, 'EDR': 81.3, 'NDR': 63.2, 'XDR': 98.0}},
    {'user_id': 'user_INS_001', 'attack_type': 'Admin Insider', 'subtype': 'Planned Exfiltration',
     'true_label': 'Attack 2', 'scores': {'SIEM': 65.1, 'EDR': 20.8, 'NDR': 82.8, 'XDR': 98.5}},
    {'user_id': 'user_INS_002', 'attack_type': 'Admin Insider', 'subtype': 'Opportunistic',
     'true_label': 'Attack 2', 'scores': {'SIEM': 75.7, 'EDR': 82.6, 'NDR': 79.0, 'XDR': 99.7}},
    {'user_id': 'user_INS_003', 'attack_type': 'Admin Insider', 'subtype': 'Gradual Buildup',
     'true_label': 'Attack 2', 'scores': {'SIEM': 51.1, 'EDR': 73.4, 'NDR': 63.3, 'XDR': 91.8}},
]


def _csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_total_events():
    path = os.path.join(DATA_DIR, 'unified_log.csv')
    if os.path.exists(path):
        size = os.path.getsize(path)
        return f'~{size // 150:,}'
    return '~750,000'


def get_attacker_scores():
    path = os.path.join(DATA_DIR, 'attacker_scores.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if data:
                return data, True
        except Exception:
            pass
    return ATTACKER_SCORES_FALLBACK, False


def get_overview_data():
    return {
        'metrics':               FINAL_METRICS,
        'experiment_data':       EXPERIMENT_DATA,
        'total_events':          get_total_events(),
        'model_comparison_full': MODEL_COMPARISON_FULL,
    }


def get_attackers_data():
    scores, is_real = get_attacker_scores()
    for atk in scores:
        uid = atk['user_id']
        atk['timeline'] = ATTACKER_TIMELINE.get(uid, {
            'days_active': 7, 'xdr_day': 1,
            'siem_day': None, 'edr_day': None, 'ndr_day': None,
        })
        atk['evidence'] = SCORE_EVIDENCE.get(uid, {
            'SIEM': '—', 'EDR': '—', 'NDR': '—', 'XDR': '—',
        })
    return {'attacker_scores': scores, 'scores_are_real': is_real}


def get_features_data():
    return {'features': FEATURES_DATA}


def get_matrices_data():
    return {'matrices': CONFUSION_MATRICES}


def get_methodology_data():
    return {
        'noise_robustness': NOISE_ROBUSTNESS,
        'learning_curve':   LEARNING_CURVE,
        'cv_scores':        CV_SCORES,
        'held_out':         HELD_OUT,
        'model_comparison': MODEL_COMPARISON_FULL,
    }


# ── Attack Story: Planned Exfiltration ────────────────────────────────────────
# alice = Admin insider (attacker)  |  bob = Busy admin (false-positive target)
# Source: pipeline_demo/scenarios/attack2_planned_exfil/ + main.py Model 06 run

PLANNED_EXFIL_STORY = {
    'timeline': [
        {'time': '08:18', 'user': 'bob',   'layer': 'SIEM', 'key': False,
         'event': 'Login success — HQ_Office (10.0.4.55)'},
        {'time': '08:22', 'user': 'bob',   'layer': 'EDR',  'key': False,
         'event': 'powershell.exe started'},
        {'time': '08:42', 'user': 'bob',   'layer': 'EDR',  'key': False,
         'event': 'psexec.exe started — patch deployment tool'},
        {'time': '09:08', 'user': 'bob',   'layer': 'NDR',  'key': False,
         'event': 'Download from 52.184.220.1 (Microsoft update server)'},
        {'time': '09:11', 'user': 'bob',   'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\IT\\patches\\KB5034441'},
        {'time': '09:14', 'user': 'alice', 'layer': 'SIEM', 'key': False,
         'event': 'Login success — HQ_Office (10.0.4.21)'},
        {'time': '09:18', 'user': 'alice', 'layer': 'EDR',  'key': False,
         'event': 'powershell.exe started'},
        {'time': '09:42', 'user': 'alice', 'layer': 'SIEM', 'key': False,
         'event': 'Failed login ×2 (typing errors)'},
        {'time': '09:42', 'user': 'bob',   'layer': 'SIEM', 'key': True,
         'event': 'Failed login ×2 — same count as Alice ← SIEM cannot distinguish'},
        {'time': '11:14', 'user': 'alice', 'layer': 'SIEM', 'key': False,
         'event': 'Privilege escalation granted'},
        {'time': '11:14', 'user': 'bob',   'layer': 'SIEM', 'key': False,
         'event': 'Privilege escalation granted'},
        {'time': '11:18', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\HR\\salaries_2024  ← category 1 of 5'},
        {'time': '11:18', 'user': 'bob',   'layer': 'NDR',  'key': False,
         'event': 'Download from 40.83.220.5 (Azure patch repo)'},
        {'time': '11:38', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\Finance\\budgets_2024  ← category 2 of 5'},
        {'time': '11:48', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\Engineering\\source_code_archive  ← category 3 of 5'},
        {'time': '12:18', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\Legal\\contracts_2024  ← category 4 of 5'},
        {'time': '12:28', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\Executive\\board_minutes  ← category 5 of 5'},
        {'time': '13:42', 'user': 'bob',   'layer': 'NDR',  'key': False,
         'event': 'Download from 52.184.220.1 (patch sync, 68 KB)'},
        {'time': '14:48', 'user': 'alice', 'layer': 'EDR',  'key': True,
         'event': '7z.exe — C:\\Windows\\Temp\\final_archive created'},
        {'time': '14:55', 'user': 'alice', 'layer': 'NDR',  'key': True,
         'event': 'data_transfer 2.284 GB → 159.65.230.11 (external VPS)'},
        {'time': '17:18', 'user': 'alice', 'layer': 'SIEM', 'key': False,
         'event': 'Session end'},
        {'time': '17:42', 'user': 'bob',   'layer': 'SIEM', 'key': False,
         'event': 'Session end — normal workday complete'},
    ],
    'feature_rows': [
        {'name': 'failed_logins_count',              'layer': 'SIEM', 'alice': '2',              'bob': '2',                'overlap': True},
        {'name': 'new_ip_flag',                      'layer': 'SIEM', 'alice': '0',              'bob': '0',                'overlap': True},
        {'name': 'login_hour',                       'layer': 'SIEM', 'alice': '9',              'bob': '8',                'overlap': True},
        {'name': 'files_vs_role_threshold',          'layer': 'EDR',  'alice': '0.575',          'bob': '0.600',            'overlap': True},
        {'name': 'file_access_spike',                'layer': 'EDR',  'alice': '1.102',          'bob': '1.024',            'overlap': True},
        {'name': 'admin_tools_used_count',           'layer': 'EDR',  'alice': '27',             'bob': '7',                'overlap': False},
        {'name': 'bytes_vs_role_threshold',          'layer': 'NDR',  'alice': '1.142',          'bob': '0.000',            'overlap': False},
        {'name': 'distinct_hosts_accessed',          'layer': 'NDR',  'alice': '2',              'bob': '2',                'overlap': True},
        {'name': 'dst_ip_category',                  'layer': 'NDR',  'alice': '2 (suspicious)', 'bob': '1 (cloud)',        'overlap': False},
        {'name': 'login_hour_deviation',             'layer': 'XDR',  'alice': '0.0',            'bob': '0.0',              'overlap': True},
        {'name': 'bytes_vs_personal_baseline',       'layer': 'XDR',  'alice': '2.285',          'bob': '0.001',            'overlap': False},
        {'name': 'file_to_transfer_gap_mins',        'layer': 'XDR',  'alice': '103.9',          'bob': '— (no transfer)',  'overlap': False},
        {'name': 'sensitive_file_category_deviation','layer': 'XDR',  'alice': '0.950',          'bob': '0.150',            'overlap': False},
        {'name': 'role_based_access_score',          'layer': 'XDR',  'alice': '0.800',          'bob': '0.218',            'overlap': False},
    ],
    'verdicts': {
        'alice': {
            'SIEM': {'pred': 2, 'conf': 0.47, 'label': 'Attack-2', 'status': 'correct'},
            'EDR':  {'pred': 0, 'conf': 0.98, 'label': 'Normal',   'status': 'fn'},
            'NDR':  {'pred': 0, 'conf': 0.98, 'label': 'Normal',   'status': 'fn'},
            'XDR':  {'pred': 2, 'conf': 0.71, 'label': 'Attack-2', 'status': 'correct'},
        },
        'bob': {
            'SIEM': {'pred': 2, 'conf': 0.46, 'label': 'Attack-2', 'status': 'fp'},
            'EDR':  {'pred': 0, 'conf': 0.99, 'label': 'Normal',   'status': 'correct'},
            'NDR':  {'pred': 0, 'conf': 0.99, 'label': 'Normal',   'status': 'correct'},
            'XDR':  {'pred': 0, 'conf': 1.00, 'label': 'Normal',   'status': 'correct'},
        },
    },
    'evidence': {
        'alice': {
            'SIEM': 'failed_logins=2 · new_ip=0 · login_hour=9 → same SIEM fingerprint as a busy admin; triggers Attack-2 at low 47% confidence',
            'EDR':  'files_vs_role=0.575 · spike=1.1 · admin_tools=27 → file count within admin range; 5-department sweep is invisible without cross-layer context',
            'NDR':  'bytes_vs_role=1.142 · hosts=2 · dst_cat=2 → above-threshold external transfer, but 3-feature NDR model returns Normal without endpoint signal',
            'XDR':  'role_based_access_score=0.80 + cat_deviation=0.95 (5 categories) + bytes_vs_personal=2.285× + gap=104 min → cross-layer chain confirms ATTACK-2',
        },
        'bob': {
            'SIEM': 'failed_logins=2 · new_ip=0 · login_hour=8 → identical SIEM profile to Alice; incorrectly flagged as Attack-2 at 46% confidence',
            'EDR':  'files_vs_role=0.6 · admin_tools=7 · spike=1.0 → all within normal IT admin range; correctly cleared',
            'NDR':  'bytes_vs_role=0.0 · hosts=2 · dst_cat=1 (mainstream cloud) → patch downloads from Microsoft/Azure only; correctly cleared',
            'XDR':  'role_based_access_score=0.218 + cat_dev=0.15 (IT only) + bytes_vs_personal=0.001 + gap=9999 (no outbound transfer) → all cross-layer signals normal; correctly cleared',
        },
    },
}


def get_story_planned_exfil():
    return PLANNED_EXFIL_STORY


# ── Attack Story: Credential Stuffing ─────────────────────────────────────────
# carol = Attack-1 (outsider using compromised IT account)
# dave  = noisy_auth — working from home, forgot VPN password 15 times
# Source: pipeline_demo/scenarios/attack1_credential_stuffing/ + run_cred_stuffing.py
# Key finding: SIEM & NDR score <70% (below alert threshold), EDR=100% Normal.
# Only XDR fires at 92%. Takeaway: daytime cred-stuffing evades 3 of 4 layers.

CRED_STUFFING_STORY = {
    'timeline': [
        {'time': '08:42', 'user': 'dave',  'layer': 'SIEM', 'key': False,
         'event': 'Failed login #1 from Home_VPN (10.0.22.44) — forgot password'},
        {'time': '09:04', 'user': 'dave',  'layer': 'SIEM', 'key': False,
         'event': 'Failed login #2 — tries again after coffee break'},
        {'time': '10:01', 'user': 'dave',  'layer': 'SIEM', 'key': False,
         'event': 'Failed logins #5–8 — still locked out'},
        {'time': '11:00', 'user': 'carol', 'layer': 'SIEM', 'key': True,
         'event': '9 automated failed logins in 6 min from 185.220.101.42 (external VPS)'},
        {'time': '11:06', 'user': 'carol', 'layer': 'SIEM', 'key': True,
         'event': 'Login success — same external IP, session established'},
        {'time': '11:10', 'user': 'carol', 'layer': 'EDR',  'key': True,
         'event': 'mimikatz.exe launched  ← credential harvesting tool'},
        {'time': '11:15', 'user': 'carol', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\IT\\config\\domain_credentials  ← category 1: IT'},
        {'time': '11:17', 'user': 'carol', 'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\HR\\employee_directory  ← category 2: HR'},
        {'time': '11:20', 'user': 'carol', 'layer': 'EDR',  'key': True,
         'event': '8 files accessed across IT + HR in 5 min  ← rapid sweep'},
        {'time': '11:29', 'user': 'dave',  'layer': 'SIEM', 'key': True,
         'event': 'Failed login #15 from Home_VPN — same new_ip=1, login_hour=11 as Carol'},
        {'time': '11:32', 'user': 'dave',  'layer': 'SIEM', 'key': False,
         'event': 'Login success — finally in; begins normal support work'},
        {'time': '11:36', 'user': 'dave',  'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\dave\\Documents\\support_ticket_8821'},
        {'time': '12:30', 'user': 'carol', 'layer': 'EDR',  'key': True,
         'event': '7z.exe — C:\\Windows\\Temp\\data_pack archive created'},
        {'time': '12:35', 'user': 'carol', 'layer': 'NDR',  'key': True,
         'event': 'data_transfer 680 MB → 185.220.101.42  ← same IP as login'},
        {'time': '14:42', 'user': 'dave',  'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\dave\\Projects\\customer_followup — normal work'},
        {'time': '15:28', 'user': 'dave',  'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\dave\\Documents\\meeting_notes — end of day'},
    ],
    'feature_rows': [
        {'name': 'failed_logins_count',              'layer': 'SIEM', 'carol': '9',              'dave': '15',              'overlap': False},
        {'name': 'new_ip_flag',                      'layer': 'SIEM', 'carol': '1',              'dave': '1',               'overlap': True},
        {'name': 'login_hour',                       'layer': 'SIEM', 'carol': '11',             'dave': '11',              'overlap': True},
        {'name': 'files_vs_role_threshold',          'layer': 'EDR',  'carol': '0.320',          'dave': '0.625',           'overlap': False},
        {'name': 'file_access_spike',                'layer': 'EDR',  'carol': '1.762',          'dave': '1.011',           'overlap': False},
        {'name': 'admin_tools_used_count',           'layer': 'EDR',  'carol': '11',             'dave': '0',               'overlap': False},
        {'name': 'bytes_vs_role_threshold',          'layer': 'NDR',  'carol': '0.680',          'dave': '0.000',           'overlap': False},
        {'name': 'distinct_hosts_accessed',          'layer': 'NDR',  'carol': '3',              'dave': '2',               'overlap': False},
        {'name': 'dst_ip_category',                  'layer': 'NDR',  'carol': '2 (suspicious)', 'dave': '1 (cloud)',       'overlap': False},
        {'name': 'login_hour_deviation',             'layer': 'XDR',  'carol': '2.0',            'dave': '2.0',             'overlap': True},
        {'name': 'bytes_vs_personal_baseline',       'layer': 'XDR',  'carol': '2.267',          'dave': '0.032',           'overlap': False},
        {'name': 'file_to_transfer_gap_mins',        'layer': 'XDR',  'carol': '74.8',           'dave': '— (no transfer)', 'overlap': False},
        {'name': 'sensitive_file_category_deviation','layer': 'XDR',  'carol': '0.600',          'dave': '0.000',           'overlap': False},
        {'name': 'role_based_access_score',          'layer': 'XDR',  'carol': '0.610',          'dave': '0.187',           'overlap': False},
    ],
    'verdicts': {
        'carol': {
            'SIEM': {'pred': 1, 'conf': 0.52, 'label': 'Attack-1', 'status': 'miss'},
            'EDR':  {'pred': 0, 'conf': 1.00, 'label': 'Normal',   'status': 'fn'},
            'NDR':  {'pred': 1, 'conf': 0.48, 'label': 'Attack-1', 'status': 'miss'},
            'XDR':  {'pred': 1, 'conf': 0.92, 'label': 'Attack-1', 'status': 'correct'},
        },
        'dave': {
            'SIEM': {'pred': 0, 'conf': 0.98, 'label': 'Normal',   'status': 'correct'},
            'EDR':  {'pred': 0, 'conf': 0.61, 'label': 'Normal',   'status': 'correct'},
            'NDR':  {'pred': 0, 'conf': 0.99, 'label': 'Normal',   'status': 'correct'},
            'XDR':  {'pred': 0, 'conf': 1.00, 'label': 'Normal',   'status': 'correct'},
        },
    },
    'evidence': {
        'carol': {
            'SIEM': 'failed=9 · new_ip=1 · login_hour=11 → overlaps with noisy_auth pattern; model scores 52% Attack-1 — below the 70% alert threshold → operational MISS',
            'EDR':  'files_vs_role=0.32 · spike=1.76 · admin_tools=11 → high admin-tool count matches normal busy-admin range; EDR has no context that these ran via mimikatz → 100% Normal — FALSE NEGATIVE',
            'NDR':  'bytes_vs_role=0.68 · hosts=3 · dst_cat=2 → suspicious destination, but 48% confidence is below the 70% alert threshold → operational MISS without cross-layer context',
            'XDR':  'role_score=0.61 + cat_dev=0.60 (IT+HR sweep) + bytes_vs_personal=2.27× + gap=75 min → cross-layer chain confirms Attack-1 at 92% — only layer above the alert threshold',
        },
        'dave': {
            'SIEM': 'failed=15 · new_ip=1 · login_hour=11 → same new_ip and login_hour as Carol, more failed logins; model scores 98% Normal because the combination does not match the attack pattern',
            'EDR':  'files_vs_role=0.625 · admin_tools=0 · spike=1.0 → all within normal support-user range; correctly cleared',
            'NDR':  'bytes_vs_role=0.0 · hosts=2 · dst_cat=1 (mainstream cloud) → Google, Azure, Dropbox only; correctly cleared',
            'XDR':  'role_score=0.187 + cat_dev=0.0 (personal files only) + bytes_vs_personal=0.032 + gap=9999 (no transfer) → all cross-layer signals normal; correctly cleared at 100%',
        },
    },
}


def get_story_cred_stuffing():
    return CRED_STUFFING_STORY


# ── Attack Story: Phishing ────────────────────────────────────────────────────
# eve   = Attack-1 (Compromised Account via phishing link, employee account)
# frank = Normal traveler — business trip, hotel WiFi, logs in at same hour
# Source: pipeline_demo/scenarios/attack1_phishing/ + run_phishing.py
# Key finding: new_ip=1, login_hour=14, dst_cat=2, login_hour_deviation=5
# ALL IDENTICAL for attacker and traveler. SIEM: Normal(44%) vs Normal(51%).
# EDR: Normal(92%) on eve — full false negative. NDR: Attack-1 at 62% — below
# threshold. Only XDR fires at 100%. "NDR cannot tell a phishing IP from hotel WiFi."

PHISHING_STORY = {
    'timeline': [
        {'time': '14:00', 'user': 'eve',   'layer': 'SIEM', 'key': True,
         'event': '2 failed logins from 198.51.100.42 (External_Phishing) — phishing C2 IP'},
        {'time': '14:00', 'user': 'frank', 'layer': 'SIEM', 'key': True,
         'event': '3 failed logins from 203.0.113.55 (Hotel_WiFi) — hotel lobby terminal typos'},
        {'time': '14:08', 'user': 'eve',   'layer': 'SIEM', 'key': True,
         'event': 'Login success — same external IP, session established  ← login_hour=14'},
        {'time': '14:14', 'user': 'frank', 'layer': 'SIEM', 'key': True,
         'event': 'Login success — hotel WiFi  ← same login_hour=14 as Eve'},
        {'time': '14:11', 'user': 'eve',   'layer': 'EDR',  'key': True,
         'event': '7z.exe launched  ← packing files for exfiltration'},
        {'time': '14:15', 'user': 'eve',   'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\HR\\employee_records  ← category 1: HR'},
        {'time': '14:18', 'user': 'frank', 'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\frank\\Documents\\client_presentation'},
        {'time': '14:22', 'user': 'eve',   'layer': 'EDR',  'key': True,
         'event': 'file_access C:\\Finance\\budget_overview_q1  ← category 2: Finance'},
        {'time': '14:26', 'user': 'eve',   'layer': 'EDR',  'key': True,
         'event': '4 files across HR + Finance in 11 min  ← outside employee role'},
        {'time': '14:52', 'user': 'eve',   'layer': 'NDR',  'key': True,
         'event': 'data_transfer 68 MB → 198.51.100.42  ← same IP as login'},
        {'time': '15:44', 'user': 'frank', 'layer': 'NDR',  'key': True,
         'event': 'connection_open → 198.18.100.22 (hotel proxy)  ← dst_cat=2 same as Eve'},
        {'time': '16:02', 'user': 'frank', 'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\frank\\Projects\\q2_planning'},
        {'time': '16:02', 'user': 'frank', 'layer': 'NDR',  'key': False,
         'event': 'download_request → 198.18.100.22 (hotel proxy, 2.8 MB)  ← normal work'},
        {'time': '17:33', 'user': 'frank', 'layer': 'EDR',  'key': False,
         'event': 'file_access C:\\Users\\frank\\Reports\\weekly_update — end of work day'},
    ],
    'feature_rows': [
        {'name': 'failed_logins_count',              'layer': 'SIEM', 'eve': '2',              'frank': '3',               'overlap': False},
        {'name': 'new_ip_flag',                      'layer': 'SIEM', 'eve': '1',              'frank': '1',               'overlap': True},
        {'name': 'login_hour',                       'layer': 'SIEM', 'eve': '14',             'frank': '14',              'overlap': True},
        {'name': 'files_vs_role_threshold',          'layer': 'EDR',  'eve': '0.667',          'frank': '0.500',           'overlap': False},
        {'name': 'file_access_spike',                'layer': 'EDR',  'eve': '1.182',          'frank': '1.008',           'overlap': False},
        {'name': 'admin_tools_used_count',           'layer': 'EDR',  'eve': '2',              'frank': '0',               'overlap': False},
        {'name': 'bytes_vs_role_threshold',          'layer': 'NDR',  'eve': '0.680',          'frank': '0.000',           'overlap': False},
        {'name': 'distinct_hosts_accessed',          'layer': 'NDR',  'eve': '2',              'frank': '2',               'overlap': True},
        {'name': 'dst_ip_category',                  'layer': 'NDR',  'eve': '2 (suspicious)', 'frank': '2 (hotel proxy)', 'overlap': True},
        {'name': 'login_hour_deviation',             'layer': 'XDR',  'eve': '5.0',            'frank': '5.0',             'overlap': True},
        {'name': 'bytes_vs_personal_baseline',       'layer': 'XDR',  'eve': '3.401',          'frank': '0.142',           'overlap': False},
        {'name': 'file_to_transfer_gap_mins',        'layer': 'XDR',  'eve': '26.0',           'frank': '— (no transfer)', 'overlap': False},
        {'name': 'sensitive_file_category_deviation','layer': 'XDR',  'eve': '0.600',          'frank': '0.000',           'overlap': False},
        {'name': 'role_based_access_score',          'layer': 'XDR',  'eve': '0.803',          'frank': '0.223',           'overlap': False},
    ],
    'verdicts': {
        'eve': {
            'SIEM': {'pred': 0, 'conf': 0.44, 'label': 'Normal',   'status': 'fn'},
            'EDR':  {'pred': 0, 'conf': 0.92, 'label': 'Normal',   'status': 'fn'},
            'NDR':  {'pred': 1, 'conf': 0.62, 'label': 'Attack-1', 'status': 'miss'},
            'XDR':  {'pred': 1, 'conf': 1.00, 'label': 'Attack-1', 'status': 'correct'},
        },
        'frank': {
            'SIEM': {'pred': 0, 'conf': 0.51, 'label': 'Normal',   'status': 'correct_barely'},
            'EDR':  {'pred': 0, 'conf': 0.89, 'label': 'Normal',   'status': 'correct'},
            'NDR':  {'pred': 0, 'conf': 0.97, 'label': 'Normal',   'status': 'correct'},
            'XDR':  {'pred': 0, 'conf': 1.00, 'label': 'Normal',   'status': 'correct'},
        },
    },
    'evidence': {
        'eve': {
            'SIEM': 'failed=2 · new_ip=1 · login_hour=14 → model scores 44% Normal (56% points toward attack but split across two attack classes); no single SIEM feature exceeds noise — FALSE NEGATIVE',
            'EDR':  'files_vs_role=0.667 · spike=1.18 · admin_tools=2 → all within the normal employee range; EDR cannot see that the files are outside her role or that 7z.exe packed them for exfil — FALSE NEGATIVE',
            'NDR':  'bytes_vs_role=0.68 · hosts=2 · dst_cat=2 → suspicious destination and above-average bytes; model scores 62% Attack-1 — below the 70% alert threshold — operational MISS',
            'XDR':  'role_score=0.803 + cat_dev=0.60 (HR+Finance sweep) + bytes_vs_personal=3.4× + gap=26 min (file to exfil) → all four XDR signals align; model fires at 100% Attack-1',
        },
        'frank': {
            'SIEM': 'failed=3 · new_ip=1 · login_hour=14 → almost identical to Eve (7% gap in confidence); SIEM scores 51% Normal — a single extra failed login would flip this to Attack-1',
            'EDR':  'files_vs_role=0.5 · admin_tools=0 · spike=1.0 → normal employee file access spread over 3 hours; correctly cleared',
            'NDR':  'bytes_vs_role=0.0 (no data_transfer) · hosts=2 · dst_cat=2 (hotel proxy) → hotel proxy is non-mainstream IP; same dst_cat=2 as Eve — only no bytes_vs_role saves NDR from a false positive here',
            'XDR':  'role_score=0.223 + cat_dev=0.0 (personal docs only) + bytes_vs_personal=0.142 + gap=9999 (no transfer) → all cross-layer signals clean; correctly cleared at 100%',
        },
    },
}


def get_story_phishing():
    return PHISHING_STORY
