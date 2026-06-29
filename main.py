import sys
import os

from modules.identity_detector import detect_identity_incidents
from modules.endpoint_detector import detect_endpoint_incidents
from modules.network_detector  import detect_network_incidents
from modules.event_loader      import EventLoader
from modules.normalizer        import Normalizer
from modules.attack_comparison import run_attack_comparison


class SuppressOutput:
    """Context manager to silence any print() calls inside a block."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._stdout


def header(title):
    """Print a stage header in a consistent style."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print()
    print("=" * 70)
    print("  XDR PIPELINE -- End-to-End Detection Walkthrough")
    print("  Author: Essam | Graduation Project 2026")
    print("=" * 70)

    # ----------------------------------------------------------------------
    #  STAGE 1: COLLECT -- Multi-Layer Telemetry Ingestion
    # ----------------------------------------------------------------------
    header("STAGE 1: COLLECT  --  Multi-Layer Telemetry Ingestion")

    required_files = ['auth_logs.csv', 'endpoint_logs.csv', 'network_logs.csv']
    for f in required_files:
        path = os.path.join("data", f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1_000_000
            print(f"  [OK] data/{f}  ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] data/{f}")
            sys.exit(1)

    with SuppressOutput():
        loader = EventLoader(data_dir="data")
        events = loader.load_all_logs()

    identity_events = [e for e in events if e.layer == "identity"]
    endpoint_events = [e for e in events if e.layer == "endpoint"]
    network_events  = [e for e in events if e.layer == "network"]

    print()
    print(f"  Layer 1 (SIEM/Identity)  : {len(identity_events):>10,} events")
    print(f"  Layer 2 (EDR/Endpoint)   : {len(endpoint_events):>10,} events")
    print(f"  Layer 3 (NDR/Network)    : {len(network_events):>10,} events")
    print(f"  " + "-" * 45)
    print(f"  Total telemetry          : {len(events):>10,} events")

    # Sample of raw events from each layer
    print()
    print("  Sample identity events (first 5):")
    for e in identity_events[:5]:
        print(f"    {e}")
    print()
    print("  Sample endpoint events (first 5):")
    for e in endpoint_events[:5]:
        print(f"    {e}")
    print()
    print("  Sample network events (first 5):")
    for e in network_events[:5]:
        print(f"    {e}")

    # ----------------------------------------------------------------------
    #  STAGE 2: NORMALIZE -- Unified Event Schema
    # ----------------------------------------------------------------------
    header("STAGE 2: NORMALIZE  --  Unified Event Schema")

    with SuppressOutput():
        normalizer = Normalizer()
        normalizer.normalize_and_save(events, path="data/unified_log.csv")
    print(f"  [OK] All {len(events):,} events normalized into shared schema")
    print(f"  [OK] Saved -> data/unified_log.csv")

    # Sample of the unified schema
    import pandas as pd
    unified = pd.read_csv("data/unified_log.csv")
    print()
    print(f"  Unified schema columns: {list(unified.columns)}")
    print()
    print("  Sample normalized rows (first 5):")
    print(unified.head(5).to_string(index=False))

    # ----------------------------------------------------------------------
    #  STAGE 3: DETECT -- Per-Layer Rule-Based Detection
    # ----------------------------------------------------------------------
    header("STAGE 3: DETECT  --  Per-Layer Rule-Based Detection")

    print("  What each single-layer tool would see in isolation:")
    print()

    print("  [SIEM-only] Identity layer (brute force / spray)...")
    with SuppressOutput():
        identity_incidents = detect_identity_incidents(identity_events)
    print(f"    -> {len(identity_incidents)} identity incidents")

    print()
    print("  [EDR-only ] Endpoint layer (sensitive file access)...")
    with SuppressOutput():
        endpoint_incidents = detect_endpoint_incidents(endpoint_events)
    print(f"    -> {len(endpoint_incidents)} endpoint incidents")

    print()
    print("  [NDR-only ] Network layer (data exfiltration)...")
    with SuppressOutput():
        network_incidents = detect_network_incidents(network_events)
    print(f"    -> {len(network_incidents)} network incidents")

    print()
    print("  Note: each single-layer tool produces alerts but cannot link them.")
    print("        The same attacker may trigger all 3 -- but no single layer knows.")

    # ----------------------------------------------------------------------
    #  STAGE 4: CORRELATE -- Cross-Layer Attack Identification
    # ----------------------------------------------------------------------
    header("STAGE 4: CORRELATE  --  Cross-Layer Attack Identification")

    print("  Linking per-layer alerts into entity-level cases via XDR correlation...")
    print()
    with SuppressOutput():
        xdr_report = run_attack_comparison(data_dir="data")

    print(f"  -> {len(xdr_report.verdicts)} XDR verdicts produced")
    print()
    print("  Confirmed attackers (first 10):")
    for v in xdr_report.verdicts[:10]:
        print(f"    {v}")

    # ----------------------------------------------------------------------
    #  STAGE 5: CLASSIFY -- Machine-Learning-Based Decision Layer
    # ----------------------------------------------------------------------
    header("STAGE 5: CLASSIFY  --  Machine-Learning-Based Decision Layer")

    print("  Final ML evaluation results (Model 06 -- XGBoost, tuned via GridSearchCV)")
    print("  See: model_06_xgboost.ipynb for full experiment + 6 anti-memorization tests")
    print()
    print(f"  {'Model':<25} {'Test F1':>10} {'Gap':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'SIEM (Identity only)':<25} {0.558:>10.3f} {0.014:>10.3f}")
    print(f"  {'EDR  (Endpoint only)':<25} {0.751:>10.3f} {0.008:>10.3f}")
    print(f"  {'NDR  (Network only)':<25} {0.625:>10.3f} {0.013:>10.3f}")
    print(f"  {'XDR  (All 14 layers)':<25} {0.948:>10.3f} {0.019:>10.3f}")
    print(f"  {'Isolation Forest':<25} {0.745:>10.3f} {'-':>10}")
    print()
    print("  Structural advantage: XDR beats best single layer (EDR) by +0.197 F1")
    print("  Confirmed across 6 experiments + 2 algorithms (RF + XGBoost)")

    # ----------------------------------------------------------------------
    #  PIPELINE COMPLETE -- Summary
    # ----------------------------------------------------------------------
    header("PIPELINE COMPLETE  --  Summary")

    print(f"  Events processed         : {len(events):,}")
    print(f"  Identity incidents       : {len(identity_incidents)}")
    print(f"  Endpoint incidents       : {len(endpoint_incidents)}")
    print(f"  Network incidents        : {len(network_incidents)}")
    print(f"  XDR verdicts             : {len(xdr_report.verdicts)}")
    print()
    print("  Output files:")
    print("    data/unified_log.csv              -- normalized telemetry")
    print("    data/xdr_correlation_results.csv  -- XDR verdicts")
    print()


if __name__ == "__main__":
    main()