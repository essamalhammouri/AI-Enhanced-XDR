"""
XDR Pipeline — Phishing Scenario
=================================
eve   : Attack-1 (Compromised Account) via phishing.
        Outsider clicks phishing link, uses stolen employee creds to log in
        at 14:08 from 198.51.100.42, sweeps HR+Finance files, exfiltrates 68 MB.

frank : Normal traveler on a business trip.
        Logs in from hotel WiFi (203.0.113.55) at 14:14 after 3 typing errors.
        Works from his laptop all afternoon through the hotel proxy.
        No suspicious activity — hotel proxy routes through non-mainstream IPs.

Key overlap: both have new_ip=1, login_hour=14, login_hour_deviation=5,
             distinct_hosts=2, AND dst_ip_category=2 (hotel proxy looks
             just as suspicious as the attacker's VPS to NDR).

The takeaway: NDR cannot tell a phishing C2 IP from a hotel WiFi proxy.
Only XDR's cross-layer features (HR+Finance file sweep + 68MB exfil to
same external IP + 26-min gap) confirm eve while clearing frank.

Run from project root:
    cd pipeline_demo
    python run_phishing.py
"""

import os, sys, csv, joblib, pandas as pd

HERE         = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(HERE, "scenarios", "attack1_phishing")
MODELS_DIR   = os.path.join(HERE, "models")

ROLE_FILE_THRESHOLDS = {
    "admin": 40, "it": 25, "manager": 15, "support": 8, "employee": 6,
}
ROLE_BYTE_THRESHOLDS = {
    "admin": 2_000_000_000, "it": 1_000_000_000,
    "manager": 200_000_000, "support": 100_000_000, "employee": 100_000_000,
}
USER_ROLES         = {"eve": "employee", "frank": "employee"}
USER_BYTE_BASELINE = {"eve": 20_000_000, "frank": 20_000_000}
USER_TYPICAL_HOUR  = {"eve": 9,          "frank": 9}

CLASS_LABELS = {0: "Normal", 1: "Attack-1 (Compromised)", 2: "Attack-2 (Insider)"}


def generate_logs():
    os.makedirs(SCENARIO_DIR, exist_ok=True)

    # ── identity_logs.csv ────────────────────────────────────────────────────
    id_rows = [["timestamp", "user", "src_ip", "location", "event", "status"]]
    id_rows.extend([
        # eve: 2 failed logins from phishing C2 IP, then success at 14:08
        ["2024-04-02 14:00:11", "eve", "198.51.100.42", "External_Phishing", "login_attempt", "failed"],
        ["2024-04-02 14:04:33", "eve", "198.51.100.42", "External_Phishing", "login_attempt", "failed"],
        ["2024-04-02 14:08:07", "eve", "198.51.100.42", "External_Phishing", "login_attempt", "success"],
        ["2024-04-02 14:08:09", "eve", "198.51.100.42", "External_Phishing", "session_start", "success"],
        # frank: 3 failed logins from hotel WiFi, success at 14:14 — same IP category and login hour
        ["2024-04-02 14:00:44", "frank", "203.0.113.55", "Hotel_WiFi", "login_attempt", "failed"],
        ["2024-04-02 14:06:18", "frank", "203.0.113.55", "Hotel_WiFi", "login_attempt", "failed"],
        ["2024-04-02 14:11:52", "frank", "203.0.113.55", "Hotel_WiFi", "login_attempt", "failed"],
        ["2024-04-02 14:14:29", "frank", "203.0.113.55", "Hotel_WiFi", "login_attempt", "success"],
        ["2024-04-02 14:14:31", "frank", "203.0.113.55", "Hotel_WiFi", "session_start", "success"],
    ])
    with open(os.path.join(SCENARIO_DIR, "identity_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(id_rows)

    # ── endpoint_logs.csv ────────────────────────────────────────────────────
    ep_rows = [["timestamp", "host", "user", "event", "file", "process", "privilege_level"]]
    ep_rows.extend([
        # eve: packs files with 7z, then accesses HR+Finance (2 sensitive categories)
        # 4 files in ~11 minutes → moderate spike; ft=6 → files_vs_role=0.667
        ["2024-04-02 14:11:08", "GW-EVE",    "eve", "process_start", "",
         "7z.exe",        "user"],
        ["2024-04-02 14:15:00", "GW-EVE",    "eve", "file_access",
         r"C:\HR\employee_records",          "explorer.exe", "user"],
        ["2024-04-02 14:18:00", "GW-EVE",    "eve", "file_access",
         r"C:\HR\salary_matrix_2024",        "explorer.exe", "user"],
        ["2024-04-02 14:22:00", "GW-EVE",    "eve", "file_access",
         r"C:\Finance\budget_overview_q1",   "explorer.exe", "user"],
        ["2024-04-02 14:26:00", "GW-EVE",    "eve", "file_access",
         r"C:\Finance\quarterly_targets_2024","explorer.exe","user"],
        ["2024-04-02 14:51:00", "GW-EVE",    "eve", "file_create",
         r"C:\Windows\Temp\report_pack",     "7z.exe",       "user"],
        # frank: 3 files spread over ~3 hours — normal traveler working from laptop
        ["2024-04-02 14:18:44", "LAPTOP-055", "frank", "file_access",
         r"C:\Users\frank\Documents\client_presentation", "explorer.exe", "user"],
        ["2024-04-02 16:02:11", "LAPTOP-055", "frank", "file_access",
         r"C:\Users\frank\Projects\q2_planning",          "explorer.exe", "user"],
        ["2024-04-02 17:33:28", "LAPTOP-055", "frank", "file_access",
         r"C:\Users\frank\Reports\weekly_update",         "explorer.exe", "user"],
    ])
    with open(os.path.join(SCENARIO_DIR, "endpoint_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(ep_rows)

    # ── network_logs.csv ─────────────────────────────────────────────────────
    nw_rows = [["timestamp", "host", "src_ip", "dest_ip", "dest_port",
                "protocol", "bytes_sent", "event", "user"]]
    nw_rows.extend([
        # eve: auth then 68 MB exfil to 198.51.100.42 (non-mainstream → dst_cat=2)
        # 2 source hosts (EXT-AUTH, GW-EVE); bytes_vs_role=0.68; bytes_personal=3.4
        ["2024-04-02 14:00:11", "EXT-AUTH", "198.51.100.42", "10.0.0.5",
         "443", "HTTPS", "4892",      "auth_request",    "eve"],
        ["2024-04-02 14:08:07", "EXT-AUTH", "198.51.100.42", "10.0.0.5",
         "443", "HTTPS", "4892",      "auth_request",    "eve"],
        ["2024-04-02 14:12:18", "GW-EVE",   "198.51.100.42", "198.51.100.42",
         "443", "HTTPS", "8241",      "connection_open", "eve"],
        ["2024-04-02 14:52:00", "GW-EVE",   "198.51.100.42", "198.51.100.42",
         "443", "HTTPS", "68000000",  "data_transfer",   "eve"],
        # frank: auth + hotel proxy traffic to 198.18.100.22 (non-mainstream → dst_cat=2!)
        # Same dst_cat as eve — NDR cannot distinguish hotel proxy from phishing C2
        # 2 source hosts (EXT-AUTH, LAPTOP-055)
        ["2024-04-02 14:00:44", "EXT-AUTH",   "203.0.113.55", "10.0.0.5",
         "443", "HTTPS", "4218",    "auth_request",    "frank"],
        ["2024-04-02 14:18:44", "LAPTOP-055", "203.0.113.55", "198.18.100.22",
         "443", "HTTPS", "15000",   "connection_open",  "frank"],
        ["2024-04-02 15:44:11", "LAPTOP-055", "203.0.113.55", "198.18.100.22",
         "443", "HTTPS", "8000",    "connection_open",  "frank"],
        ["2024-04-02 16:02:11", "LAPTOP-055", "203.0.113.55", "198.18.100.22",
         "443", "HTTPS", "2800000", "download_request", "frank"],
        ["2024-04-02 17:00:33", "LAPTOP-055", "203.0.113.55", "198.18.100.22",
         "443", "HTTPS", "6000",    "connection_open",  "frank"],
    ])
    with open(os.path.join(SCENARIO_DIR, "network_logs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(nw_rows)

    print(f"  Wrote logs to {SCENARIO_DIR}")


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
    failed     = ((u_id["event"] == "login_attempt") & (u_id["status"] == "failed")).sum()
    new_ip     = int(u_id["location"].iloc[0] not in ("HQ_Office",)) if len(u_id) else 0
    ok         = u_id[(u_id["event"] == "login_attempt") & (u_id["status"] == "success")]
    login_hour = pd.to_datetime(ok["timestamp"].iloc[0]).hour if len(ok) else 9

    # EDR
    files         = (u_ep["event"] == "file_access").sum()
    files_vs_role = round(files / ft, 3) if ft else 0
    spike = 1.0
    ts = pd.to_datetime(u_ep[u_ep["event"] == "file_access"]["timestamp"])
    if len(ts) > 1:
        density = len(ts) / max(1, (ts.max() - ts.min()).total_seconds() / 60)
        spike   = round(min(5.0, density * 0.5 + 1.0), 3)
    admin_tools = u_ep["process"].isin(
        ["powershell.exe", "psexec.exe", "7z.exe", "mimikatz.exe",
         "certutil.exe", "net.exe"]
    ).sum()

    # NDR
    outbound       = u_nw[u_nw["event"] == "data_transfer"]["bytes_sent"].sum()
    bytes_vs_role  = round(outbound / bt, 3) if bt else 0
    distinct_hosts = u_nw["host"].nunique()
    external       = u_nw[~u_nw["dest_ip"].str.startswith("10.")]["dest_ip"]
    mainstream     = ("52.", "40.", "142.250", "151.101", "162.125", "8.8")
    if len(external) == 0:
        dst_cat = 0
    elif external.apply(lambda ip: any(ip.startswith(p) for p in mainstream)).all():
        dst_cat = 1
    else:
        dst_cat = 2

    # XDR-only
    hour_dev       = round(abs(login_hour - typical), 3)
    total_bytes    = u_nw["bytes_sent"].sum()
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
        "failed_logins_count":               int(failed),
        "new_ip_flag":                       int(new_ip),
        "login_hour":                        int(login_hour),
        "files_vs_role_threshold":           float(files_vs_role),
        "file_access_spike":                 float(spike),
        "admin_tools_used_count":            int(admin_tools),
        "bytes_vs_role_threshold":           float(bytes_vs_role),
        "distinct_hosts_accessed":           int(distinct_hosts),
        "dst_ip_category":                   int(dst_cat),
        "login_hour_deviation":              float(hour_dev),
        "bytes_vs_personal_baseline":        float(bytes_personal),
        "file_to_transfer_gap_mins":         float(gap),
        "sensitive_file_category_deviation": float(cat_dev),
        "role_based_access_score":           float(role_score),
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
    print("  PHISHING PIPELINE — eve (attacker) vs frank (traveler)")
    print("="*72)

    generate_logs()

    identity = pd.read_csv(os.path.join(SCENARIO_DIR, "identity_logs.csv"))
    endpoint = pd.read_csv(os.path.join(SCENARIO_DIR, "endpoint_logs.csv"))
    network  = pd.read_csv(os.path.join(SCENARIO_DIR, "network_logs.csv"))
    print(f"\n  Loaded: {len(identity)} identity | {len(endpoint)} endpoint "
          f"| {len(network)} network events")

    feats = {u: compute_features(u, identity, endpoint, network) for u in USER_ROLES}

    siem = ["failed_logins_count", "new_ip_flag", "login_hour"]
    edr  = ["files_vs_role_threshold", "file_access_spike", "admin_tools_used_count"]
    ndr  = ["bytes_vs_role_threshold", "distinct_hosts_accessed", "dst_ip_category"]
    xdr  = ["login_hour_deviation", "bytes_vs_personal_baseline",
            "file_to_transfer_gap_mins", "sensitive_file_category_deviation",
            "role_based_access_score"]

    print(f"\n  {'feature':<42}{'eve':>10}{'frank':>10}")
    print("  " + "-"*62)
    for k in siem: print(f"  [SIEM] {k:<35}{feats['eve'][k]:>10}{feats['frank'][k]:>10}")
    for k in edr:  print(f"  [EDR ] {k:<35}{feats['eve'][k]:>10}{feats['frank'][k]:>10}")
    for k in ndr:  print(f"  [NDR ] {k:<35}{feats['eve'][k]:>10}{feats['frank'][k]:>10}")
    for k in xdr:  print(f"  [XDR ] {k:<35}{feats['eve'][k]:>10}{feats['frank'][k]:>10}")

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
