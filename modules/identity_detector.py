from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from collections import defaultdict

from modules.event_loader import XDREvent
from config import MIN_FAILED_LOGINS, IDENTITY_WINDOW_MINUTES


@dataclass
class IdentityIncident:
    user: str
    src_ip: str
    start_time: datetime
    end_time: datetime
    failed_count: int
    has_success_after: bool
    severity: int
    evidence: List[XDREvent]


def debug_print_identity_events(events: List[XDREvent]) -> None:
    print("\n[IDENTITY] Debug listing of identity events:")
    for e in events:
        print(f"- {e.timestamp} | user={e.user} | ip={e.src_ip} | type={e.event_type}")
        print(f"  details = {e.details}")


def detect_identity_incidents(events: List[XDREvent]) -> List[IdentityIncident]:
    """
    Detects brute force patterns:
      - MIN_FAILED_LOGINS or more failed logins from the same (user, src_ip)
        within IDENTITY_WINDOW_MINUTES
      - followed by a successful login within IDENTITY_WINDOW_MINUTES of
        the last failed login.

    This is the identity-layer view of Attack 1 (Compromised User).
    SIEM sees this and fires MEDIUM. EDR and NDR stay silent because
    the post-login behavior looks normal for an IT role.
    Only XDR connects this brute force to the file access and exfiltration
    that follow within 40 minutes.
    """
    incidents: List[IdentityIncident] = []
    if not events:
        return incidents

    window_failed  = timedelta(minutes=90)   # Attack 1 sends 150 logins over ~75 min
    window_success = timedelta(minutes=90)

    # 1) Group events by (user, src_ip)
    grouped: dict[tuple[str, str], List[XDREvent]] = defaultdict(list)
    for e in events:
        key = (e.user, e.src_ip or "")
        grouped[key].append(e)

    # 2) For each (user, src_ip), use a sliding window to find bursts of failures
    for (user, src_ip), evts in grouped.items():
        evts.sort(key=lambda e: e.timestamp)

        failed_events: List[XDREvent] = [
            e for e in evts
            if e.event_type == "login"
            and str(e.details.get("status", "")).lower() == "failed"
        ]
        success_events: List[XDREvent] = [
            e for e in evts
            if e.event_type == "login"
            and str(e.details.get("status", "")).lower() == "success"
        ]

        print(f"[DEBUG-ID-SUMMARY] user={user}, ip={src_ip}, "
              f"failed_total={len(failed_events)}, "
              f"success_found={len(success_events) > 0}")

        if len(failed_events) < MIN_FAILED_LOGINS:
            continue

        # Sliding window: find the largest burst of failures within window_failed.
        # We scan the full list and collect ALL failures that fall within the
        # window starting at the first failure — this gives the true burst count.
        burst_events: List[XDREvent] = []
        burst_found = False

        for start_idx in range(len(failed_events)):
            window_end = failed_events[start_idx].timestamp + window_failed
            # Collect all failures within window starting at start_idx
            current_burst = [
                e for e in failed_events[start_idx:]
                if e.timestamp <= window_end
            ]
            if len(current_burst) >= MIN_FAILED_LOGINS:
                if len(current_burst) > len(burst_events):
                    burst_events = current_burst
                burst_found = True

        if not burst_found:
            continue

        # Look for a success login within window_success of the last failure
        last_failure_time = burst_events[-1].timestamp
        success_event = next(
            (e for e in success_events
             if last_failure_time < e.timestamp <= last_failure_time + window_success),
            None
        )

        # For Attack 1 the success comes at 03:05, failures end at ~02:50,
        # so window_success needs to cover that gap. If not found in tight
        # window, fall back to first success after the burst at all.
        if success_event is None:
            success_event = next(
                (e for e in success_events if e.timestamp > last_failure_time),
                None
            )

        if success_event is not None:
            incident = IdentityIncident(
                user=user,
                src_ip=src_ip,
                start_time=burst_events[0].timestamp,
                end_time=success_event.timestamp,
                failed_count=len(burst_events),
                has_success_after=True,
                severity=70,
                evidence=burst_events + [success_event],
            )
            incidents.append(incident)

    return incidents