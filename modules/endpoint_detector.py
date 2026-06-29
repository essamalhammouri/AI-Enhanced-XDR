from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List
from collections import defaultdict

from modules.event_loader import XDREvent
from config import ENDPOINT_WINDOW_MINUTES, ROLE_SENSITIVE_FILE_THRESHOLDS

SENSITIVE_FILES = [
    "confidential_report.docx",
    "finance_data.xlsx",
    "employee_records.csv",
    "secret_project.pdf",
    "salary_info.xlsx",
    "strategic_plan.docx",
]

# Attack 1 — Compromised User: user_477=it, user_478=it, user_479=admin
# Must be listed explicitly so they get their real role thresholds.
COMPROMISED_USER_ROLES = {"user_477": "it", "user_478": "it", "user_479": "admin"}


def get_role(user: str) -> str:
    """Derive role string from username prefix."""
    if user in COMPROMISED_USER_ROLES: return COMPROMISED_USER_ROLES[user]
    if user.startswith("admin_"):      return "admin"
    if user.startswith("it_"):         return "it"
    if user.startswith("traveler_"):   return "traveler"
    if user.startswith("noisy_"):      return "noisy"
    # Attack 2 — Silent Insider roles (all threshold=8)
    if user == "user_474": return "support"
    if user == "user_475": return "assistant"
    if user == "user_476": return "junior"
    return "normal"


@dataclass
class EndpointIncident:
    user: str
    host: str
    start_time: datetime
    end_time: datetime
    sensitive_files_accessed: List[str]
    process: str
    severity: int
    evidence: List[XDREvent]
    role: str = "normal"
    role_threshold: int = 8


def detect_endpoint_incidents(events: List[XDREvent]) -> List[EndpointIncident]:
    """
    Detects bulk sensitive file access, respecting per-role baselines.

    WHY role-aware thresholds matter:
    - admin_* and it_* legitimately access many sensitive files daily.
      A global threshold would flood alerts with false positives.
      Role thresholds silence those.

    Attack 1 — Compromised User (user_471/472/473):
      - Role=it → threshold=25. They access 20 files → UNDER threshold.
      - EDR stays completely SILENT.
      - XDR needed to connect brute force → file access → exfiltration.

    Attack 2 — Silent Insider (silent_001/002/003):
      - Role=normal → threshold=8. They access 10 files → OVER threshold.
      - EDR fires MEDIUM.
      - XDR escalates to CRITICAL: 0 files in 30-day baseline + 2am login
        + 10 files + 800KB exfil 8 minutes later, all in one session.
    """
    incidents: List[EndpointIncident] = []
    if not events:
        return incidents

    # Group events by (user, host)
    grouped: dict[tuple, List[XDREvent]] = defaultdict(list)
    for e in events:
        grouped[(e.user, e.host)].append(e)

    for (user, host), evts in grouped.items():
        evts.sort(key=lambda e: e.timestamp)

        role = get_role(user)
        threshold = ROLE_SENSITIVE_FILE_THRESHOLDS.get(role, 8)

        # Collect sensitive file access events for this user+host
        sensitive_hits: List[XDREvent] = [
            e for e in evts
            if e.event_type == "file_access"
            and str(e.details.get("file", "")).lower() in SENSITIVE_FILES
        ]

        if len(sensitive_hits) >= threshold:
            incidents.append(EndpointIncident(
                user=user,
                host=host,
                start_time=sensitive_hits[0].timestamp,
                end_time=sensitive_hits[-1].timestamp,
                sensitive_files_accessed=[
                    str(e.details.get("file", "")) for e in sensitive_hits
                ],
                process=str(sensitive_hits[0].details.get("process", "")),
                severity=60,
                evidence=sensitive_hits,
                role=role,
                role_threshold=threshold,
            ))

    return incidents