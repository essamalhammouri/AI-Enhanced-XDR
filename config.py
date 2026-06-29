# config.py

# --- Identity Detection -------------------------------------------------------
MIN_FAILED_LOGINS = 10          # noisy_* max out at 5 — this keeps them silent
IDENTITY_WINDOW_MINUTES = 60   # Attack 1 sends 100 logins over ~50 min

# --- Endpoint Detection -------------------------------------------------------
MIN_SENSITIVE_FILES = 2          # global fallback (not used directly anymore)
ENDPOINT_WINDOW_MINUTES = 10

# --- Network Detection --------------------------------------------------------
MIN_BYTES_SENT = 500_000         # global fallback (not used directly anymore)
INTERNAL_IP_PREFIX = "192.168."
NETWORK_WINDOW_MINUTES = 10

# --- Role-Based Behavioral Baselines -----------------------------------------
# Endpoint: how many sensitive file accesses before we flag a user
# Higher threshold = more tolerance = fewer false positives for privileged roles
ROLE_SENSITIVE_FILE_THRESHOLDS = {
    "admin":    40,   # admins legitimately access many sensitive files
    "it":       25,   # IT staff do too, just less
    "normal":    8,   # baseline for regular users
    "traveler":  8,
    "noisy":     8,
    # Attack 2 — Silent Insider roles (low-privilege, threshold=8)
    # 10 files > 8 → EDR fires MEDIUM → XDR escalates to CRITICAL
    "support":   8,
    "assistant": 8,
    "junior":    8,
    # Attack 1 — Compromised User: user_471/472/473 use role=it (threshold=25)
    # so 20 files stays UNDER threshold → EDR silent → XDR needed
}

# Network: bytes sent to suspicious IPs before we flag a host
ROLE_BYTES_THRESHOLDS = {
    "admin":     2_000_000_000,   # admins push large legitimate payloads (patches, images)
    "it":        1_000_000_000,   # IT staff do software pushes, log uploads
    # Attack 1 — Compromised User: small exfil stays UNDER threshold → NDR silent
    "normal":      100_000_000,
    "traveler":    100_000_000,
    "noisy":       100_000_000,
    # Attack 2 — Silent Insider: targeted exfil stays UNDER threshold → NDR silent
    "support":     100_000_000,
    "assistant":   100_000_000,
    "junior":      100_000_000,
}
# --- Correlation Engine -------------------------------------------------------
CORRELATION_WINDOW_MINUTES = 20

# --- AI Scorer ----------------------------------------------------------------
SEVERITY_HIGH_THRESHOLD   = 80
SEVERITY_MEDIUM_THRESHOLD = 50