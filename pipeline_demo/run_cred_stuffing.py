"""
XDR Pipeline — Credential Stuffing Scenario
============================================
carol : Attack-1 (Compromised Account) via credential stuffing.
        Outsider tries 87 passwords against carol's IT account over 47 min
        (02:11–02:58), succeeds at 03:01, then exfiltrates ~1 MB.
dave  : Normal noisy_auth user. Forgot his password 8 times during the
        morning, eventually logged in, did routine support work all day.

Designed so SIEM cannot cleanly separate them (both have failed logins
from the same calendar day). XDR's cross-layer features resolve the case.

Run from project root:
    cd pipeline_demo
    python run_cred_stuffing.py
"""

import os
import sys
import csv
import joblib
import pandas as pd

HERE        = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(HERE, "scenarios", "attack1_credential_stuffing")
MODELS_DIR  = os.path.join(HERE, "models")

# ── User configs ────────────────────────────────────────────────────────────
ROLE_FILE_THRESHOLDS = {
    "admin": 40, "it": 25, "manager": 15, "support": 8, "employee": 6,
}
ROLE_BYTE_THRESHOLDS = {
    "admin": 2_000_000_000, "it": 1_000_000_000,
    "manager": 200_000_000, "support": 100_000_000, "employee": 100_000_000,
}
USER_ROLES         = {"carol": "it",          "dave": "support"}
USER_BYTE_BASELINE = {"carol": 300_000_000,  "dave": 2_000_000}
USER_TYPICAL_HOUR  = {"carol": 9,             "dave": 9}

CLASS_LABELS = {0: "Normal", 1: "Attack-1 (Compromised)", 2: "Attack-2 (Insider)"}

# ── Log generation ──────────────────────────────────────────────────────────

def generate_logs():
    os.makedirs(SCENARIO_DIR, exist_ok=True)

    # ── identity_logs.csv ──────────────────────────────────────────────────
    id_rows = [["timestamp", "user", "src_ip", "location", "event", "status"]]

    # carol: 9 failed logins then success at 11:06 AM — daytime attack to blend in
    # Training-distribution: cred_stuffing login_hour center = 11, failed=[1,14], new_ip=1
    import datetime
    t = datetime.datetime(2024, 3, 15, 11, 0, 14)
    for _ in range(9):
        id_rows.append([t.strftime("%Y-%m-%d %H:%M:%S"), "carol",
                        "185.220.101.42", "External_VPN", "login_attempt", "failed"])
        t += datetime.timedelta(seconds=36)
    id_rows.append(["2024-03-15 11:06:08", "carol", "185.220.101.42",
                    "External_VPN", "login_attempt", "success"])
    id_rows.append(["2024-03-15 11:06:10", "carol", "185.220.101.42",
                    "External_VPN", "session_start", "success"])

    # dave: 15 failed logins from Home_VPN (new IP) — working from home, forgot password
    # new_ip_flag=1 because location != "HQ_Office"; logs in at 11 AM same as carol
    dave_times = [
        "08:42:11", "09:04:33", "09:22:07", "09:44:52",
        "10:01:18", "10:18:43", "10:33:28", "10:47:05",
        "10:58:41", "11:08:14", "11:14:48", "11:19:33",
        "11:23:07", "11:26:42", "11:29:18",
    ]
    for ts in dave_times:
        id_rows.append([f"2024-03-15 {ts}", "dave", "10.0.22.44",
                        "Home_VPN", "login_attempt", "failed"])
    id_rows.append(["2024-03-15 11:32:44", "dave", "10.0.22.44",
                    "Home_VPN", "login_attempt", "success"])
    id_rows.append(["2024-03-15 11:32:46", "dave", "10.0.22.44",
                    "Home_VPN", "session_start", "success"])

    with open(os.path.join(SCENARIO_DIR, "identity_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(id_rows)

    # ── endpoint_logs.csv ──────────────────────────────────────────────────
    ep_rows = [["timestamp", "host", "user", "event", "file", "process", "privilege_level"]]
    ep_rows.extend([
        # carol: 8 file accesses across IT+HR in 6 minutes — rapid post-login sweep
        # + 2 admin tools (mimikatz, 7z) — 2 sensitive categories → cat_dev=0.60
        ["2024-03-15 11:10:11", "TS-REMOTE", "carol", "process_start", "", "mimikatz.exe", "admin"],
        ["2024-03-15 11:15:00", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\config\domain_credentials",    "mimikatz.exe", "admin"],
        ["2024-03-15 11:15:45", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\admin_tools\service_accounts", "mimikatz.exe", "admin"],
        ["2024-03-15 11:16:30", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\backup_configs\ad_backup",     "powershell.exe", "admin"],
        ["2024-03-15 11:17:15", "TS-REMOTE", "carol", "file_access",
         r"C:\HR\employee_directory",           "powershell.exe", "admin"],
        ["2024-03-15 11:18:00", "TS-REMOTE", "carol", "file_access",
         r"C:\HR\salary_bands",                 "powershell.exe", "admin"],
        ["2024-03-15 11:18:45", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\security\auth_logs",           "powershell.exe", "admin"],
        ["2024-03-15 11:19:30", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\config\vpn_profiles",          "powershell.exe", "admin"],
        ["2024-03-15 11:20:15", "TS-REMOTE", "carol", "file_access",
         r"C:\IT\scripts\user_provisioning",    "powershell.exe", "admin"],
        ["2024-03-15 12:30:08", "TS-REMOTE", "carol", "process_start", "", "7z.exe", "admin"],
        ["2024-03-15 12:31:22", "TS-REMOTE", "carol", "file_create",
         r"C:\Windows\Temp\data_pack",          "7z.exe", "admin"],
        # dave: 5 files in personal/support area — no sensitive categories at all
        ["2024-03-15 11:36:11", "WS-055", "dave", "file_access",
         r"C:\Users\dave\Documents\support_ticket_8821", "explorer.exe", "user"],
        ["2024-03-15 13:18:33", "WS-055", "dave", "file_access",
         r"C:\Users\dave\Documents\q1_report_draft",     "explorer.exe", "user"],
        ["2024-03-15 14:02:07", "WS-055", "dave", "file_access",
         r"C:\Users\dave\Templates\response_template",   "explorer.exe", "user"],
        ["2024-03-15 14:42:44", "WS-055", "dave", "file_access",
         r"C:\Users\dave\Projects\customer_followup",    "explorer.exe", "user"],
        ["2024-03-15 15:28:19", "WS-055", "dave", "file_access",
         r"C:\Users\dave\Documents\meeting_notes_march", "explorer.exe", "user"],
    ])
    with open(os.path.join(SCENARIO_DIR, "endpoint_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(ep_rows)

    # ── network_logs.csv ───────────────────────────────────────────────────
    nw_rows = [["timestamp", "host", "src_ip", "dest_ip", "dest_port",
                "protocol", "bytes_sent", "event", "user"]]
    nw_rows.extend([
        # carol: auth bursts then 680 MB transfer to C2 (185.x — suspicious)
        # 3 source hosts → distinct_hosts=3; bytes_vs_personal=2.27; gap ~70 min
        ["2024-03-15 11:00:14", "EXT-AUTH",  "185.220.101.42", "10.0.0.5",
         "443", "HTTPS", "4892",      "auth_request",    "carol"],
        ["2024-03-15 11:06:08", "EXT-AUTH",  "185.220.101.42", "10.0.0.5",
         "443", "HTTPS", "4892",      "auth_request",    "carol"],
        ["2024-03-15 11:12:18", "TS-REMOTE", "185.220.101.42", "185.220.101.42",
         "443", "HTTPS", "8241",      "connection_open", "carol"],
        ["2024-03-15 12:35:00", "GW-EXT",    "185.220.101.42", "185.220.101.42",
         "443", "HTTPS", "680000000", "data_transfer",   "carol"],
        # dave: mainstream cloud only — no data_transfer, dst_cat=1
        ["2024-03-15 11:32:44", "EXT-AUTH",  "10.0.22.44", "10.0.0.5",
         "443", "HTTPS", "4218",  "auth_request",    "dave"],
        ["2024-03-15 11:35:04", "WS-055",    "10.0.22.44", "142.250.80.14",
         "443", "HTTPS", "28244", "connection_open",  "dave"],
        ["2024-03-15 13:18:11", "WS-055",    "10.0.22.44", "40.83.220.5",
         "443", "HTTPS", "8821",  "connection_open",  "dave"],
        ["2024-03-15 14:42:07", "WS-055",    "10.0.22.44", "162.125.2.18",
         "443", "HTTPS", "12442", "download_request", "dave"],
        ["2024-03-15 15:28:19", "WS-055",    "10.0.22.44", "142.250.80.14",
         "443", "HTTPS", "9814",  "connection_open",  "dave"],
    ])
    with open(os.path.join(SCENARIO_DIR, "network_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(nw_rows)

    print(f"  Wrote logs to {SCENARIO_DIR}")


# ── Feature extraction (same logic as main.py, new user configs) ─────────────

def compute_features(user, identity, endpoint, network):
    role     = USER_ROLES[user]
    ft       = ROLE_FILE_THRESHOLDS[role]
    bt       = ROLE_BYTE_THRESHOLDS[role]
    baseline = USER_BYTE_BASELINE[user]
    typical  = USER_TYPICAL_HOUR[user]

    u_id = identity[identity["user"] == user]
    u_ep = endpoint[endpoint["user"] == user]
    u_nw = network[network["user"] == user]

    # SIEM
    failed = ((u_id["event"] == "login_attempt") & (u_id["status"] == "failed")).sum()
    new_ip = int(u_id["location"].iloc[0] not in ("HQ_Office",)) if len(u_id) else 0
    ok = u_id[(u_id["event"] == "login_attempt") & (u_id["status"] == "success")]
    login_hour = pd.to_datetime(ok["timestamp"].iloc[0]).hour if len(ok) else 9

    # EDR
    files = (u_ep["event"] == "file_access").sum()
    files_vs_role = round(files / ft, 3) if ft else 0
    spike = 1.0
    ts = pd.to_datetime(u_ep[u_ep["event"] == "file_access"]["timestamp"])
    if len(ts) > 1:
        density = len(ts) / max(1, (ts.max() - ts.min()).total_seconds() / 60)
        spike = round(min(5.0, density * 0.5 + 1.0), 3)
    admin_tools = u_ep["process"].isin(
        ["powershell.exe", "psexec.exe", "7z.exe", "mimikatz.exe",
         "certutil.exe", "net.exe"]
    ).sum()

    # NDR
    outbound = u_nw[u_nw["event"] == "data_transfer"]["bytes_sent"].sum()
    bytes_vs_role = round(outbound / bt, 3) if bt else 0
    distinct_hosts = u_nw["host"].nunique()
    external = u_nw[~u_nw["dest_ip"].str.startswith("10.")]["dest_ip"]
    mainstream = ("52.", "40.", "142.250", "151.101", "162.125", "8.8")
    if len(external) == 0:
        dst_cat = 0
    elif external.apply(lambda ip: any(ip.startswith(p) for p in mainstream)).all():
        dst_cat = 1
    else:
        dst_cat = 2

    # XDR-only
    hour_dev = round(abs(login_hour - typical), 3)
    total_bytes = u_nw["bytes_sent"].sum()
    bytes_personal = round(total_bytes / baseline, 3) if baseline else 0

    last_file  = u_ep[u_ep["event"] == "file_access"]["timestamp"]
    first_xfer = u_nw[u_nw["event"] == "data_transfer"]["timestamp"]
    if len(last_file) > 0 and len(first_xfer) > 0:
        gap = round(abs((pd.to_datetime(first_xfer).min() -
                         pd.to_datetime(last_file).max()).total_seconds()) / 60, 1)
    else:
        gap = 9999.0

    cats = set()
    for f in u_ep[u_ep["event"] == "file_access"]["file"].astype(str):
        fl = f.lower()
        if "engineering" in fl: cats.add("Engineering")
        if "confidential" in fl: cats.add("Confidential")
        if "hr" in fl: cats.add("HR")
        if "finance" in fl: cats.add("Finance")
        if "legal" in fl: cats.add("Legal")
        if "sales" in fl: cats.add("Sales")
        if "executive" in fl or "merger" in fl: cats.add("Executive")
        if "\\it\\" in fl or fl.startswith("c:\\it\\"): cats.add("IT")
    if len(cats) <= 1:
        cat_dev = round(0.15 * len(cats), 3)
    else:
        cat_dev = round(min(0.95, 0.40 + 0.20 * (len(cats) - 1)), 3)

    files_norm   = min(1.0, files / max(1, ft * 0.6))
    bytes_factor = min(1.0, bytes_personal / 3.0)
    hour_factor  = min(1.0, hour_dev / 6.0)
    role_score   = round(min(0.95,
        0.45 * cat_dev + 0.30 * bytes_factor +
        0.15 * files_norm + 0.10 * hour_factor), 3)

    return {
        "failed_logins_count":              int(failed),
        "new_ip_flag":                      int(new_ip),
        "login_hour":                       int(login_hour),
        "files_vs_role_threshold":          float(files_vs_role),
        "file_access_spike":                float(spike),
        "admin_tools_used_count":           int(admin_tools),
        "bytes_vs_role_threshold":          float(bytes_vs_role),
        "distinct_hosts_accessed":          int(distinct_hosts),
        "dst_ip_category":                  int(dst_cat),
        "login_hour_deviation":             float(hour_dev),
        "bytes_vs_personal_baseline":       float(bytes_personal),
        "file_to_transfer_gap_mins":        float(gap),
        "sensitive_file_category_deviation": float(cat_dev),
        "role_based_access_score":          float(role_score),
    }


def load_models():
    files = {"SIEM": "siem_model.pkl", "EDR": "edr_model.pkl",
             "NDR": "ndr_model.pkl",  "XDR": "xdr_model.pkl"}
    loaded = {}
    for short, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found"); sys.exit(1)
        loaded[short] = joblib.load(path)
    return loaded


def predict(model_payload, feats):
    order  = model_payload["features"]
    vector = [feats[name] for name in order]
    df     = pd.DataFrame([vector], columns=order)
    pred   = int(model_payload["model"].predict(df)[0])
    proba  = model_payload["model"].predict_proba(df)[0]
    return pred, proba


def main():
    print("\n" + "="*72)
    print("  CREDENTIAL STUFFING PIPELINE — carol (attacker) vs dave (noisy_auth)")
    print("="*72)

    generate_logs()

    identity = pd.read_csv(os.path.join(SCENARIO_DIR, "identity_logs.csv"))
    endpoint = pd.read_csv(os.path.join(SCENARIO_DIR, "endpoint_logs.csv"))
    network  = pd.read_csv(os.path.join(SCENARIO_DIR, "network_logs.csv"))
    print(f"\n  Loaded: {len(identity)} identity | {len(endpoint)} endpoint "
          f"| {len(network)} network events")

    feats = {u: compute_features(u, identity, endpoint, network)
             for u in USER_ROLES}

    siem = ["failed_logins_count", "new_ip_flag", "login_hour"]
    edr  = ["files_vs_role_threshold", "file_access_spike", "admin_tools_used_count"]
    ndr  = ["bytes_vs_role_threshold", "distinct_hosts_accessed", "dst_ip_category"]
    xdr  = ["login_hour_deviation", "bytes_vs_personal_baseline",
            "file_to_transfer_gap_mins", "sensitive_file_category_deviation",
            "role_based_access_score"]

    print(f"\n  {'feature':<42}{'carol':>10}{'dave':>10}")
    print("  " + "-"*62)
    for k in siem:
        print(f"  [SIEM] {k:<35}{feats['carol'][k]:>10}{feats['dave'][k]:>10}")
    for k in edr:
        print(f"  [EDR ] {k:<35}{feats['carol'][k]:>10}{feats['dave'][k]:>10}")
    for k in ndr:
        print(f"  [NDR ] {k:<35}{feats['carol'][k]:>10}{feats['dave'][k]:>10}")
    for k in xdr:
        print(f"  [XDR ] {k:<35}{feats['carol'][k]:>10}{feats['dave'][k]:>10}")

    models = load_models()

    print(f"\n  {'user':<8}{'SIEM':>22}{'EDR':>22}{'NDR':>22}{'XDR':>22}")
    print("  " + "-"*95)
    verdicts = {}
    for user in feats:
        verdicts[user] = {}
        line = f"  {user:<8}"
        for short in ["SIEM", "EDR", "NDR", "XDR"]:
            pred, proba = predict(models[short], feats[user])
            conf = proba[pred]
            verdicts[user][short] = (pred, conf)
            label = CLASS_LABELS[pred].split(" ")[0]
            cell  = f"{label[:8]} ({conf:.2f})"
            line += f"{cell:>22}"
        print(line)

    print("\n  Summary:")
    for user, vd in verdicts.items():
        row = f"  {user:<8}"
        for short in ["SIEM", "EDR", "NDR", "XDR"]:
            pred, conf = vd[short]
            tag = "Normal" if pred == 0 else ("ATK-1" if pred == 1 else "ATK-2")
            row += f"  {short}={tag}({conf:.2f})"
        print(row)


if __name__ == "__main__":
    main()
