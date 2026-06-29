from typing import List, Dict
import pandas as pd
from modules.event_loader import XDREvent

SEVERITY_MAP = {
    # Identity (auth_logs)
    "login":                0,
    "successful_login":     0,
    "failed_login":         2,
    "login_outside_hours":  3,
    "mfa_bypass":           4,
    # Endpoint (endpoint_logs)
    "process_exec":         1,
    "file_access":          1,
    "privilege_escalation": 4,
    "lateral_movement":     4,
    "ransomware_behavior":  5,
    "usb_insert":           2,
    # Network (network_logs)
    "outbound_connection":  1,
    "data_exfiltration":    5,
    "port_scan":            4,
    "c2_beacon":            5,
}

class Normalizer:
    def normalize(self, events: List[XDREvent]) -> List[Dict]:
        normalized = []
        for e in events:
            normalized.append({
                "timestamp": e.timestamp.isoformat(),
                "user":       e.user,
                "machine":    e.host,           # renamed to match CSV schema
                "src_ip":     e.src_ip or "",
                "dst_ip":     e.dst_ip or "",
                "event_type": e.event_type,
                "layer":      e.layer,
                "severity":   SEVERITY_MAP.get(e.event_type, 1),  # <- NEW
            })
        return normalized

    def normalize_and_save(self, events: List[XDREvent], path="data/unified_log.csv"):
        records = self.normalize(events)
        df = pd.DataFrame(records)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path, index=False)
        print(f"unified_log.csv -> {len(df):,} rows")
        return df