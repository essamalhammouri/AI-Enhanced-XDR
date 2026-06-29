"""
XDR Pipeline Demonstration (defense version)
=============================================
Loads the 4 trained Model 06 XGBoost classifiers from disk and runs them
on two hand-designed user-windows:

  alice : attack2_planned_exfil. Admin abusing privileges to exfiltrate
          across HR, Finance, Engineering, Legal, Executive categories.
  bob   : normal_busy_admin. Admin doing legitimate patch deployment work.

Both users are admin/IT roles at the office during work hours. Their
single-layer features overlap heavily (busy_admin is designed to overlap
with attack 2 in the dataset). Single-layer ML models cannot reliably
tell them apart -- this is why Model 06's EDR-only F1 caps around 0.74.

XDR's full 14-feature model uses cross-layer signals
(role_based_access_score, file_to_transfer_gap_mins,
bytes_vs_personal_baseline, sensitive_file_category_deviation) that
no single layer can compute alone. XDR catches alice and clears bob.

Run from project root:
    cd pipeline_demo
    python main.py
"""

import os
import sys
import joblib
import pandas as pd

# ---------- CONFIG ---------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(HERE, "scenarios", "attack2_planned_exfil")
MODELS_DIR   = os.path.join(HERE, "models")

ROLE_FILE_THRESHOLDS = {
    "admin": 40, "it": 25, "manager": 15, "support": 8, "employee": 6,
}
ROLE_BYTE_THRESHOLDS = {
    "admin": 2_000_000_000, "it": 1_000_000_000, "manager": 200_000_000,
    "support": 100_000_000, "employee": 100_000_000,
}
USER_ROLES         = {"alice": "admin", "bob": "admin"}
USER_BYTE_BASELINE = {"alice": 1_000_000_000, "bob": 1_000_000_000}
USER_TYPICAL_HOUR  = {"alice": 9, "bob": 8}

CLASS_LABELS = {0: "Normal", 1: "Attack-1 (Compromised)", 2: "Attack-2 (Insider)"}

WIDTH = 76


# ---------- helpers --------------------------------------------------------

def hr(): print("=" * WIDTH)

def stage_header(num, title):
    print(); hr(); print(f"  STAGE {num} -- {title}"); hr()

def section(title):
    print(); print(f"  -- {title}")

def small_sample(df, n=3):
    for _, row in df.head(n).iterrows():
        line = "     " + " | ".join(str(v)[:24] for v in row.values)
        print(line[:WIDTH + 4])


# ---------- STAGE 1 --------------------------------------------------------

def stage1_collect():
    stage_header(1, "COLLECT (raw log ingestion)")
    print("  XDR collects telemetry from three security layers:")
    print("     - Identity layer (SIEM)")
    print("     - Endpoint layer (EDR)")
    print("     - Network  layer (NDR)")

    identity = pd.read_csv(os.path.join(SCENARIO_DIR, "identity_logs.csv"))
    endpoint = pd.read_csv(os.path.join(SCENARIO_DIR, "endpoint_logs.csv"))
    network  = pd.read_csv(os.path.join(SCENARIO_DIR, "network_logs.csv"))

    print()
    print(f"  Loaded:")
    print(f"     identity_logs.csv  -> {len(identity)} events")
    print(f"     endpoint_logs.csv  -> {len(endpoint)} events")
    print(f"     network_logs.csv   -> {len(network)} events")

    users = sorted(set(list(identity['user']) + list(endpoint['user']) + list(network['user'])))
    print(f"  Users present: {users}")

    section("Sample identity events")
    small_sample(identity[["timestamp","user","src_ip","event","status"]])
    section("Sample endpoint events")
    small_sample(endpoint[["timestamp","user","event","file","process"]])
    section("Sample network events")
    small_sample(network[["timestamp","user","dest_ip","bytes_sent","event"]])

    return identity, endpoint, network


# ---------- STAGE 2 --------------------------------------------------------

def stage2_normalize(identity, endpoint, network):
    stage_header(2, "NORMALIZE (unify three schemas into one event format)")
    print("  Map all three schemas into a unified event format:")
    print("     (timestamp, actor, layer, event_type)")

    rows = []
    for _, r in identity.iterrows():
        rows.append({"timestamp": r["timestamp"], "actor": r["user"],
                     "layer": "SIEM",
                     "event_type": f"AUTH_{r['event'].upper()}_{r['status'].upper()}"})
    for _, r in endpoint.iterrows():
        rows.append({"timestamp": r["timestamp"], "actor": r["user"],
                     "layer": "EDR", "event_type": r["event"].upper()})
    for _, r in network.iterrows():
        rows.append({"timestamp": r["timestamp"], "actor": r["user"],
                     "layer": "NDR", "event_type": r["event"].upper()})

    unified = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    section("Unified event count by layer")
    for layer, n in unified["layer"].value_counts().items():
        print(f"     {layer:5s}  {n:>3d} events")
    print(f"     TOTAL  {len(unified)} events")

    section("Sample unified events (first 6)")
    small_sample(unified, n=6)
    return unified


# ---------- STAGE 3 --------------------------------------------------------

def stage3_correlate(unified):
    stage_header(3, "CORRELATE (group events into per-user windows)")
    print("  XDR's correlation engine groups events by actor. Each user")
    print("  becomes one user-window containing every event from every layer.")
    print()
    print(f"     {'user':<10}{'SIEM':>8}{'EDR':>8}{'NDR':>8}{'TOTAL':>8}")
    for user in sorted(unified["actor"].unique()):
        u = unified[unified["actor"] == user]
        siem = (u["layer"] == "SIEM").sum()
        edr  = (u["layer"] == "EDR").sum()
        ndr  = (u["layer"] == "NDR").sum()
        print(f"     {user:<10}{siem:>8}{edr:>8}{ndr:>8}{len(u):>8}")


# ---------- STAGE 4 (feature extraction) -----------------------------------

def compute_features(user, identity, endpoint, network):
    role     = USER_ROLES[user]
    ft       = ROLE_FILE_THRESHOLDS[role]
    bt       = ROLE_BYTE_THRESHOLDS[role]
    baseline = USER_BYTE_BASELINE[user]
    typical  = USER_TYPICAL_HOUR[user]

    u_id = identity[identity["user"] == user]
    u_ep = endpoint[endpoint["user"] == user]
    u_nw = network[network["user"] == user]

    # SIEM features
    failed = ((u_id["event"] == "login_attempt") & (u_id["status"] == "failed")).sum()
    new_ip = int(u_id["location"].iloc[0] not in ("HQ_Office",)) if len(u_id) else 0
    ok = u_id[(u_id["event"] == "login_attempt") & (u_id["status"] == "success")]
    login_hour = pd.to_datetime(ok["timestamp"].iloc[0]).hour if len(ok) else 9

    # EDR features
    files = (u_ep["event"] == "file_access").sum()
    files_vs_role = round(files / ft, 3) if ft else 0
    spike = 1.0
    ts = pd.to_datetime(u_ep[u_ep["event"] == "file_access"]["timestamp"])
    if len(ts) > 1:
        density = len(ts) / max(1, (ts.max() - ts.min()).total_seconds() / 60)
        spike = round(min(5.0, density * 0.5 + 1.0), 3)
    admin_tools = u_ep["process"].isin(
        ["powershell.exe","psexec.exe","7z.exe","mimikatz.exe","certutil.exe","net.exe"]
    ).sum()

    # NDR features
    outbound = u_nw[u_nw["event"] == "data_transfer"]["bytes_sent"].sum()
    bytes_vs_role = round(outbound / bt, 3) if bt else 0
    distinct_hosts = u_nw["host"].nunique()
    external = u_nw[~u_nw["dest_ip"].str.startswith("10.")]["dest_ip"]
    mainstream = ("52.","40.","142.250","151.101","162.125","8.8")
    if len(external) == 0:
        dst_cat = 0
    elif external.apply(lambda ip: any(ip.startswith(p) for p in mainstream)).all():
        dst_cat = 1
    else:
        dst_cat = 2

    # XDR-only features
    hour_dev = round(abs(login_hour - typical), 3)
    total_bytes = u_nw["bytes_sent"].sum()
    bytes_personal = round(total_bytes / baseline, 3) if baseline else 0

    last_file = u_ep[u_ep["event"] == "file_access"]["timestamp"]
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

    files_norm = min(1.0, files / max(1, ft * 0.6))
    bytes_factor = min(1.0, bytes_personal / 3.0)
    hour_factor = min(1.0, hour_dev / 6.0)

    role_score = round(min(0.95,
        0.45 * cat_dev + 0.30 * bytes_factor +
        0.15 * files_norm + 0.10 * hour_factor), 3)

    return {
        "failed_logins_count": int(failed),
        "new_ip_flag": int(new_ip),
        "login_hour": int(login_hour),
        "files_vs_role_threshold": float(files_vs_role),
        "file_access_spike": float(spike),
        "admin_tools_used_count": int(admin_tools),
        "bytes_vs_role_threshold": float(bytes_vs_role),
        "distinct_hosts_accessed": int(distinct_hosts),
        "dst_ip_category": int(dst_cat),
        "login_hour_deviation": float(hour_dev),
        "bytes_vs_personal_baseline": float(bytes_personal),
        "file_to_transfer_gap_mins": float(gap),
        "sensitive_file_category_deviation": float(cat_dev),
        "role_based_access_score": float(role_score),
    }


def stage4_features(identity, endpoint, network):
    stage_header(4, "FEATURE EXTRACTION (compute the 14 ML features)")
    print("  9 single-layer features come first.")
    print("  5 XDR-only features are DERIVED from cross-layer correlation.")

    feats = {u: compute_features(u, identity, endpoint, network) for u in USER_ROLES}

    siem = ["failed_logins_count","new_ip_flag","login_hour"]
    edr  = ["files_vs_role_threshold","file_access_spike","admin_tools_used_count"]
    ndr  = ["bytes_vs_role_threshold","distinct_hosts_accessed","dst_ip_category"]
    xdr  = ["login_hour_deviation","bytes_vs_personal_baseline",
            "file_to_transfer_gap_mins","sensitive_file_category_deviation",
            "role_based_access_score"]

    section("9 single-layer features")
    print(f"     {'feature':<40}{'alice':>10}{'bob':>10}")
    print("     " + "-" * 60)
    for k in siem:
        print(f"     [SIEM] {k:<33}{feats['alice'][k]:>10}{feats['bob'][k]:>10}")
    for k in edr:
        print(f"     [EDR ] {k:<33}{feats['alice'][k]:>10}{feats['bob'][k]:>10}")
    for k in ndr:
        print(f"     [NDR ] {k:<33}{feats['alice'][k]:>10}{feats['bob'][k]:>10}")

    section("5 XDR-only cross-layer features")
    print(f"     {'feature':<42}{'alice':>9}{'bob':>9}")
    print("     " + "-" * 60)
    for k in xdr:
        print(f"     [XDR ] {k:<35}{feats['alice'][k]:>9}{feats['bob'][k]:>9}")

    return feats


# ---------- STAGE 5 (single-layer ML classification) -----------------------

def load_models():
    """Load the 4 trained Model 06 XGBoost models from disk."""
    files = {
        "SIEM":    "siem_model.pkl",
        "EDR":     "edr_model.pkl",
        "NDR":     "ndr_model.pkl",
        "XDR":     "xdr_model.pkl",
    }
    loaded = {}
    for short, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.isfile(path):
            print(f"\nERROR: model file not found: {path}")
            print("Run the save-models cell in model_06_xgboost.ipynb first.")
            sys.exit(1)
        loaded[short] = joblib.load(path)
    return loaded


def predict_for_user(model_payload, feats):
    """Build a feature vector in the model's expected order, predict."""
    order = model_payload["features"]
    vector = [feats[name] for name in order]
    df = pd.DataFrame([vector], columns=order)
    pred = int(model_payload["model"].predict(df)[0])
    proba = model_payload["model"].predict_proba(df)[0]
    return pred, proba


def stage5_classify(feats, models):
    stage_header(5, "CLASSIFY (4 trained Model 06 XGBoost classifiers)")
    print("  Each model below was trained in model_06_xgboost.ipynb on the")
    print("  same 8000-row dataset. The single-layer models see only their")
    print("  own 3 features. The XDR model sees all 14.")
    print()
    print("  Loaded from pipeline_demo/models/:")
    for short in ["SIEM","EDR","NDR","XDR"]:
        m = models[short]
        n_feats = len(m["features"])
        print(f"     {m['name']:<25}  ({n_feats} features)")

    section("Predictions per user")
    print(f"     {'user':<8}{'SIEM':>22}{'EDR':>22}{'NDR':>22}{'XDR':>22}")
    print("     " + "-" * 95)

    verdicts = {}
    for user in feats:
        verdicts[user] = {}
        line = f"     {user:<8}"
        for short in ["SIEM","EDR","NDR","XDR"]:
            pred, proba = predict_for_user(models[short], feats[user])
            label = CLASS_LABELS[pred]
            conf = proba[pred]
            verdicts[user][short] = (pred, conf, label)
            cell = f"{label.split(' ')[0][:8]} ({conf:.2f})"
            line += f"{cell:>22}"
        print(line)

    section("Compact verdict table")
    print(f"     {'user':<10}{'SIEM':>10}{'EDR':>10}{'NDR':>10}{'XDR':>10}")
    print("     " + "-" * 50)
    for user, vd in verdicts.items():
        row = f"     {user:<10}"
        for short in ["SIEM","EDR","NDR","XDR"]:
            pred, _, _ = vd[short]
            verdict = "Normal" if pred == 0 else ("ATTACK-1" if pred == 1 else "ATTACK-2")
            row += f"{verdict:>10}"
        print(row)

    return verdicts


# ---------- STAGE 6 (interpretation) ---------------------------------------

def stage6_interpret(verdicts, models):
    stage_header(6, "INTERPRET (link to Model 06 test-set performance)")
    print("  The verdicts above came from the actual trained Model 06 XGBoost")
    print("  classifiers. Their test-set F1 scores are below for reference.")
    print()

    print(f"     Model              Test F1     Role")
    for short, label in [("SIEM","SIEM-only"), ("EDR","EDR-only"),
                         ("NDR","NDR-only"),  ("XDR","XDR (full)")]:
        # F1 not stored in payload; print a placeholder note instead
        print(f"     {label:<18}                  sees {len(models[short]['features'])} features")
    print()
    print("  Reference numbers from notebook (run with seed=42):")
    print("     SIEM ~0.56   EDR ~0.74   NDR ~0.64   XDR ~0.95")
    print("     XDR vs best single layer: roughly +0.20 F1.")


# ---------- final summary --------------------------------------------------

def final_summary(verdicts):
    print(); hr()
    print("  PIPELINE COMPLETE -- THESIS CLAIM MADE VISIBLE")
    hr(); print()

    def label_of(pred):
        return "Normal" if pred == 0 else ("Attack-1" if pred == 1 else "Attack-2")

    a = verdicts["alice"]; b = verdicts["bob"]
    print(f"  alice (designed as attack2_planned_exfil):")
    print(f"     SIEM model: {label_of(a['SIEM'][0])}    EDR model: {label_of(a['EDR'][0])}")
    print(f"     NDR  model: {label_of(a['NDR'][0])}    XDR model: {label_of(a['XDR'][0])}")
    print()
    print(f"  bob (designed as normal_busy_admin):")
    print(f"     SIEM model: {label_of(b['SIEM'][0])}    EDR model: {label_of(b['EDR'][0])}")
    print(f"     NDR  model: {label_of(b['NDR'][0])}    XDR model: {label_of(b['XDR'][0])}")
    print()
    print("  At every single layer, alice's behaviour overlaps with bob's:")
    print("     - both are admin role")
    print("     - both work normal hours from office IPs")
    print("     - both have high file-access counts and admin-tool usage")
    print("     - both transfer non-trivial bytes")
    print()
    print("  The cross-layer features (role_based_access_score,")
    print("  sensitive_file_category_deviation, file_to_transfer_gap_mins,")
    print("  bytes_vs_personal_baseline) are what tell them apart. These")
    print("  features cannot be computed by SIEM, EDR, or NDR alone.")


# ---------- main -----------------------------------------------------------

def main():
    print(); hr()
    print("  XDR PIPELINE DEMONSTRATION (real Model 06 classifiers)")
    print("  Scenario: planned_exfil (alice) vs busy_admin (bob)")
    hr()

    if not os.path.isdir(SCENARIO_DIR):
        print(f"\nERROR: Scenario folder not found: {SCENARIO_DIR}")
        sys.exit(1)
    if not os.path.isdir(MODELS_DIR):
        print(f"\nERROR: Models folder not found: {MODELS_DIR}")
        sys.exit(1)

    identity, endpoint, network = stage1_collect()
    unified = stage2_normalize(identity, endpoint, network)
    stage3_correlate(unified)
    feats = stage4_features(identity, endpoint, network)

    models = load_models()
    verdicts = stage5_classify(feats, models)
    stage6_interpret(verdicts, models)
    final_summary(verdicts)


if __name__ == "__main__":
    main()