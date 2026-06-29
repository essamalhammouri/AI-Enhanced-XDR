from dataclasses import dataclass
from typing import List
from collections import defaultdict
from config import ROLE_BYTES_THRESHOLDS

SUSPICIOUS_IPS = [
    "185.220.101.5",
    "45.33.32.156",
    "91.108.4.1",
    "203.0.113.50",
    "104.21.44.5",
]

# Attack 1 — Compromised User: user_477=it, user_478=it, user_479=admin
COMPROMISED_USER_ROLES = {"user_477": "it", "user_478": "it", "user_479": "admin"}


def get_role(user: str) -> str:
    """Derive role string from username prefix."""
    if user in COMPROMISED_USER_ROLES: return COMPROMISED_USER_ROLES[user]
    if user.startswith("admin_"):      return "admin"
    if user.startswith("it_"):         return "it"
    if user.startswith("traveler_"):   return "traveler"
    if user.startswith("noisy_"):      return "noisy"
    # Attack 2 — Silent Insider roles (all 1MB threshold → 800KB stays UNDER → NDR silent)
    if user == "user_474": return "support"
    if user == "user_475": return "assistant"
    if user == "user_476": return "junior"
    return "normal"


@dataclass
class NetworkIncident:
    host: str
    user: str
    dst_ip: str
    total_bytes_sent: int
    severity: str
    role: str = "normal"
    role_threshold: int = 1_000_000


def detect_network_incidents(events) -> List[NetworkIncident]:
    """
    Flags hosts sending large volumes to suspicious IPs.
    Thresholds are role-aware to eliminate false positives.

    Attack 1 — Compromised User (user_477/478/479):
      - Sends large bytes to SAFE IPs after brute force — looks like normal IT/admin work.
      - Destination is NOT suspicious → NDR never even checks it → COMPLETELY SILENT.
      - EDR also silent — 20 files is under IT/admin role threshold.
      - Only XDR catches it: brute force + file access + large transfer within 40 min.

    Attack 2 — Silent Insider (user_474/475/476):
      - Role=support/assistant/junior → threshold=1MB. Exfil 800KB → UNDER → NDR SILENT.
      - XDR catches it: 800KB = 2× personal daily baseline of 400KB,
        happening 8 minutes after sensitive file access at 2am.
    """
    # Accumulate bytes per (host, user, dst_ip) — keying on user prevents
    # bytes from different users on the same host from accumulating together.
    traffic: dict[tuple, int] = defaultdict(int)

    for e in events:
        details    = getattr(e, 'details', {}) or {}
        dst        = details.get('dest_ip') or getattr(e, 'dst_ip', None)
        bytes_sent = details.get('bytes_sent', 0) or 0
        host       = details.get('host') or getattr(e, 'host', None)
        user       = getattr(e, 'user', None) or details.get('user', 'unknown')

        if dst in SUSPICIOUS_IPS:
            key = (host, user, dst)
            traffic[key] += bytes_sent

    incidents = []
    for (host, user, dst_ip), total_bytes in traffic.items():
        role      = get_role(user)
        threshold = ROLE_BYTES_THRESHOLDS.get(role, 1_000_000)

        if total_bytes >= threshold:
            severity = "CRITICAL" if total_bytes >= 2_000_000 else "HIGH"
            incidents.append(NetworkIncident(
                host=host,
                user=user,
                dst_ip=dst_ip,
                total_bytes_sent=total_bytes,
                severity=severity,
                role=role,
                role_threshold=threshold,
            ))

    return incidents