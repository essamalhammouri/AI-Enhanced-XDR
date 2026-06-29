import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class XDREvent:
    timestamp: datetime
    layer: str
    user: str
    host: str
    event_type: str
    src_ip: Optional[str]
    dst_ip: Optional[str]
    details: dict

class EventLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_all_logs(self) -> List[XDREvent]:
        events: List[XDREvent] = []
        sources = {
            "auth_logs.csv":     "identity",
            "endpoint_logs.csv": "endpoint",
            "network_logs.csv":  "network",
        }
        for filename, layer in sources.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                print(f"[WARN] {path} not found, skipping")
                continue
            print(f"[INFO] Loading {layer} logs from {filename}")
            df = pd.read_csv(path)
            events.extend(self._parse_layer(df, layer))
        events.sort(key=lambda e: e.timestamp)
        print(f"[INFO] Loaded {len(events):,} events in total")
        return events

    def _parse_layer(self, df: pd.DataFrame, layer: str) -> List[XDREvent]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # ── Unified column aliases ────────────────────────────────────────────
        # WHY: Each CSV has different column names. We create short aliases
        # (u, h, ev, sip, dip) so the XDREvent constructor stays the same
        # regardless of which layer we're parsing.
        df["u"]  = df["user"]    if "user"    in df.columns else ""
        df["h"]  = df["machine"] if "machine" in df.columns else (df["host"] if "host" in df.columns else "")
        df["ev"] = df["event"]   if "event"   in df.columns else (df["event_type"] if "event_type" in df.columns else "")

        if layer == "identity":
            df["sip"] = df["src_ip"]  if "src_ip"  in df.columns else ""
            df["dip"] = ""
        elif layer == "endpoint":
            df["sip"] = ""
            df["dip"] = ""
        else:
            df["sip"] = df["src_ip"]  if "src_ip"  in df.columns else ""
            df["dip"] = df["dest_ip"] if "dest_ip" in df.columns else ""

        # ── Build events with full details dict ───────────────────────────────
        # WHY FIX: The old code passed details={} (empty dict) for every event.
        # This meant the rule-based detectors (identity_detector, endpoint_detector,
        # network_detector) could never read fields like 'status', 'bytes_sent',
        # 'privilege_level', 'location', 'file' — so they found 0 incidents.
        #
        # The fix: we copy ALL raw columns from the CSV row into the details dict.
        # We exclude only our internal alias columns (u, h, ev, sip, dip) and
        # Index/timestamp since those are already stored as proper fields.
        # Now e.details.get("status") returns "failed", e.details.get("bytes_sent")
        # returns the actual number, etc.
        EXCLUDE = {'Index', 'timestamp', 'u', 'h', 'ev', 'sip', 'dip'}

        events = []
        for row in df.itertuples():
            raw = row._asdict()
            details = {k: v for k, v in raw.items() if k not in EXCLUDE}
            events.append(XDREvent(
                timestamp  = row.timestamp,
                layer      = layer,
                user       = row.u,
                host       = row.h,
                event_type = row.ev,
                src_ip     = row.sip or None,
                dst_ip     = row.dip or None,
                details    = details,
            ))
        return events