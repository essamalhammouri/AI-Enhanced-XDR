"""
attack_comparison.py — Phase 7: Attack Comparison Report

The final output of the XDR pipeline. Produces four sections:

  1. Cross-layer blindness table (SIEM / EDR / NDR / XDR per attack)
  2. Minute-by-minute attack timelines
  3. UEBA baseline comparison (what XDR knew that single-layer tools didn't)
  4. Final XDR verdicts with full reasoning

This module is the proof of the academic argument:
  Single-layer tools either miss attacks or score them LOW.
  XDR correlates across layers and time to score the same attacks CRITICAL.
  The gap between single-layer scores and XDR scores is the proof.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

from modules.correlation_engine import run_correlation_engine, CorrelationReport, XDRVerdict
from modules.graph_builder import build_graph, print_graph_summary


# ── Formatting helpers ────────────────────────────────────────────────────────
SEVERITY_DISPLAY = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH":     "🟠 HIGH",
    "MEDIUM":   "🟡 MEDIUM",
    "LOW":      "🟢 LOW",
    "SILENT":   "⚪ SILENT",
}

def fmt_severity(s: str) -> str:
    return SEVERITY_DISPLAY.get(s, s)

def divider(char="=", width=70):
    return char * width

def section(title: str):
    print("\n" + divider())
    print(f"  {title}")
    print(divider())


# ── Section 1: Cross-layer blindness table ────────────────────────────────────
def print_blindness_table(report: CorrelationReport):
    section("SECTION 1 — CROSS-LAYER BLINDNESS TABLE")
    print("""
  This table shows what each single-layer tool sees vs what XDR sees.
  SILENT = tool made a correct single-layer decision but missed the attack.
  The gap between single-layer and XDR severity is the core academic proof.
""")

    header = f"  {'User':<12} {'Attack Type':<26} {'SIEM':<10} {'EDR':<10} {'NDR':<10} {'XDR':>10}"
    print(header)
    print(f"  {'-'*12} {'-'*26} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for _, row in report.summary_table.iterrows():
        xdr_display = fmt_severity(row["XDR Severity"])
        print(
            f"  {row['User']:<12} "
            f"{row['Attack'][:26]:<26} "
            f"{row['SIEM']:<10} "
            f"{row['EDR']:<10} "
            f"{row['NDR']:<10} "
            f"{xdr_display:>14}"
        )

    print(f"""
  Key takeaways:
    Attack 1 (Compromised User) : SIEM=MEDIUM, EDR=SILENT, NDR=SILENT → XDR=CRITICAL
      → EDR and NDR both made correct role-based decisions. The account looked
        legitimate. Only XDR knew it was compromised (brute force 40 min earlier).

    Attack 2 (Silent Insider)   : SIEM=SILENT, EDR=MEDIUM, NDR=SILENT → XDR=CRITICAL
      → EDR raised MEDIUM. NDR stayed silent. Neither was wrong.
        XDR escalated because: 0 files in 30 days → 10 files at 2am → 800KB
        8 minutes later. The MEDIUM alert was not a MEDIUM attack.
""")


# ── Section 2: Minute-by-minute timelines ─────────────────────────────────────
def print_attack_timelines(report: CorrelationReport):
    section("SECTION 2 — MINUTE-BY-MINUTE ATTACK TIMELINES")

    attack1_verdicts = [v for v in report.verdicts if "Attack 1" in v.attack_type]
    attack2_verdicts = [v for v in report.verdicts if "Attack 2" in v.attack_type]

    print(f"\n  ── Attack 1: Compromised User (example: {attack1_verdicts[0].user}) ──")
    print("""
  TIME   LAYER      EVENT                                   TOOL RESPONSE
  ─────  ─────────  ──────────────────────────────────────  ──────────────────
  01:00  Identity   150 failed logins from 185.220.101.5   SIEM: MEDIUM alert
                    (1 attempt every 30 seconds)            EDR:  no visibility
                                                            NDR:  no visibility
  02:20  Identity   Login SUCCESS — same suspicious IP      SIEM: brute force +
                    Attacker now has valid session           success = noted
  02:25  Endpoint   20 sensitive files accessed             EDR:  SILENT
                    (explorer.exe, role=it)                 (20 < IT threshold 25)
  03:05  Network    ~1.5–2.5MB to external IP               NDR:  SILENT
                    (203.0.113.50 — looks like IT backup)   (safe IP, normal role)
  ─────  ─────────  ──────────────────────────────────────  ──────────────────
  XDR:   brute force → file access → exfiltration in 40 min → CRITICAL
""")

    print(f"\n  ── Attack 2: Silent Insider (example: {attack2_verdicts[0].user}) ──")
    print("""
  TIME   LAYER      EVENT                                   TOOL RESPONSE
  ─────  ─────────  ──────────────────────────────────────  ──────────────────
  [30-day baseline]
         Identity   Always logs in 08:00–18:00 Jordan       SIEM: normal
         Endpoint   Zero sensitive file accesses             EDR:  normal
         Network    ~400KB/day to safe external IPs          NDR:  normal

  [Attack day — day 71/72/73]
  02:00  Identity   Login at 2am — same IP, same country    SIEM: SILENT
                    Valid credentials, no brute force        (no rule triggers)
  02:10  Endpoint   10 sensitive files accessed              EDR:  MEDIUM
                    (0 such accesses in prior 30 days)       (10 ≥ threshold 8)
  02:38  Network    800KB to 45.33.32.156 (suspicious IP)   NDR:  SILENT
                    (8 minutes after file access)            (800KB < 1MB limit)
  ─────  ─────────  ──────────────────────────────────────  ──────────────────
  XDR:   0 files in 30 days + 2am login + files + suspicious transfer → CRITICAL
""")


# ── Section 3: UEBA baseline comparison ───────────────────────────────────────
def print_ueba_baselines(report: CorrelationReport):
    section("SECTION 3 — UEBA BEHAVIORAL BASELINES")
    print("""
  This section shows what XDR knows about each user's normal behavior
  before the attack day. Single-layer tools have no per-user baseline.
  XDR compares each event to this baseline to detect deviation.
""")

    for v in report.verdicts:
        if not v.baseline_facts:
            # Attack 1 — show why baseline matters here too
            print(f"  {v.user} ({v.attack_type})")
            print(f"    Baseline notes: Normal IT/admin behavior masked the attack.")
            print(f"    Role threshold (files): 25 for IT, 40 for admin")
            print(f"    NDR threshold (bytes) : 2MB for IT, 3MB for admin")
            print(f"    Why XDR matters       : Brute force is the baseline break —")
            print(f"                            no legitimate session starts with")
            print(f"                            150 failed logins from a foreign IP.")
            print()
        else:
            print(f"  {v.user} ({v.attack_type})")
            for fact in v.baseline_facts:
                print(f"    • {fact}")
            print(f"    Attack day deviations:")
            for item in v.evidence_chain:
                if "UEBA" in item or "CHAIN" in item:
                    print(f"      {item.strip()}")
            print()


# ── Section 4: Final XDR verdicts ─────────────────────────────────────────────
def print_final_verdicts(report: CorrelationReport):
    section("SECTION 4 — FINAL XDR VERDICTS")

    for v in report.verdicts:
        print(f"\n  ┌─ {v.user}  [{v.attack_type}] ─")
        print(f"  │")

        # Single-layer scores side by side
        print(f"  │  Single-layer tool verdicts:")
        for lv in v.layer_verdicts:
            fired_label = "fired" if lv.fired else "silent"
            print(f"  │    {lv.tool:<6} → {lv.severity:<8} ({fired_label})")

        # XDR score breakdown
        print(f"  │")
        print(f"  │  XDR Correlation Score:")
        print(f"  │    Base (highest single-layer)     : {v.base_score}")
        for name, val in v.correlation_bonuses.items():
            label = name.replace("_", " ").title()
            print(f"  │    + {label:<35} +{val}")
        print(f"  │    {'─'*44}")
        print(f"  │    Final XDR Score                 : {v.final_score} / 100")
        print(f"  │    XDR Severity                    : {fmt_severity(v.severity)}")
        print(f"  │")

        # Key insight
        print(f"  │  Key Insight:")
        # Word-wrap the insight at ~60 chars
        words = v.key_insight.split()
        line = "  │    \""
        for word in words:
            if len(line) + len(word) > 72:
                print(line)
                line = "  │     " + word + " "
            else:
                line += word + " "
        print(line.rstrip() + "\"")

        # Attack pattern classification
        if v.classification:
            c = v.classification
            print(f"  │")
            print(f"  │  ▶ XDR Attack Classification:")
            print(f"  │    Detected type : {c.detected_type}")
            print(f"  │    Confidence    : {c.confidence} ({c.confidence_score}/100)")
            print(f"  │    Signals that triggered this classification:")
            for sig in c.matched_signals:
                words_s = sig.split()
                sig_line = "  │      • "
                for w in words_s:
                    if len(sig_line) + len(w) > 74:
                        print(sig_line)
                        sig_line = "  │        " + w + " "
                    else:
                        sig_line += w + " "
                print(sig_line.rstrip())
            print(f"  │    Ruled out:")
            words_r = c.ruling_out.split()
            r_line = "  │      "
            for w in words_r:
                if len(r_line) + len(w) > 74:
                    print(r_line)
                    r_line = "  │      " + w + " "
                else:
                    r_line += w + " "
            print(r_line.rstrip())

        print(f"  └{'─'*67}")


# ── Section 5: Why XDR summary ────────────────────────────────────────────────
def print_why_xdr_summary():
    section("SECTION 5 — WHY XDR: THE ACADEMIC ARGUMENT")
    print("""
  The two attacks above prove the core claim:

  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Capability      │ SIEM     │ EDR      │ NDR      │ XDR      │
  ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Brute force     │ YES      │ no       │ no       │ YES      │
  │ File threshold  │ no       │ YES      │ no       │ YES      │
  │ Byte threshold  │ no       │ no       │ YES      │ YES      │
  │ Chain (A1)      │ no       │ no       │ no       │ YES ←    │
  │ Baseline (A2)   │ no       │ no       │ no       │ YES ←    │
  │ Timing corr.    │ no       │ no       │ no       │ YES ←    │
  │ Graph path      │ no       │ no       │ no       │ YES ←    │
  └─────────────────┴──────────┴──────────┴──────────┴──────────┘

  The rows marked ← are what turn MEDIUM → CRITICAL.
  Each row is a reason a single-layer tool missed or underscored the attack.
  XDR sees all rows simultaneously, across time, per user.

  Attack 1 proof: "EDR and NDR both made the CORRECT single-layer decision.
  The account looked legitimate to both. Only XDR knew it was compromised
  because it saw the brute force 40 minutes earlier on the identity layer."

  Attack 2 proof: "EDR raised MEDIUM. NDR was silent. Neither was wrong.
  XDR raised CRITICAL because it knew both events happened 8 minutes apart
  at 2am by a user who had NEVER touched a sensitive file in 30 days."
""")


# ── Main entry point ──────────────────────────────────────────────────────────
def run_attack_comparison(data_dir: str = "data"):
    """
    Runs the full Phase 7 report. Called by main.py.
    """
    print("\n" + divider())
    print("  PHASE 7 — ATTACK COMPARISON REPORT")
    print(divider())
    print("  Proving cross-layer correlation detects what single tools miss.\n")

    # Run the correlation engine (also runs graph builder internally)
    report = run_correlation_engine(data_dir=data_dir)

    # Run all four report sections
    print_blindness_table(report)
    print_attack_timelines(report)
    print_ueba_baselines(report)
    print_final_verdicts(report)
    print_why_xdr_summary()

    # Confirmation
    output_path = os.path.join(data_dir, "xdr_correlation_results.csv")
    print(f"  [OK] xdr_correlation_results.csv saved → {output_path}")
    print()

    return report


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_attack_comparison(data_dir="data")