# AI-ISF V5 — Project Specification Mapping

This version maps the supplied **AI-Driven Identity Authentication and Fraud Prevention Platform** specification onto the existing AI-ISF application while preserving the working professional frontend and camera flow.

| Supplied requirement | AI-ISF V5 implementation |
|---|---|
| Secure password authentication | Scrypt-derived owner password hash |
| JWT sessions | JWT + SQLite session table + expiry + revocation |
| Authentication events | `authentication_events` SQLite table |
| Device/IP monitoring | Hashed device/IP identifiers |
| Login-time analysis | Behavioral risk feature |
| New device/location | Behavioral risk features |
| Failed-attempt analysis | Behavioral risk feature |
| AI anomaly detection | scikit-learn Isolation Forest |
| Hybrid scoring | Rule-based points + ML anomaly signal |
| Risk 0–100 | `risk_score` in every authentication event |
| LOW/MEDIUM/HIGH | Configurable prototype thresholds |
| ALLOW/VERIFY/BLOCK | Risk engine recommended action |
| Fraud alerts | `fraud_alerts` table + dashboard API |
| Audit logging | `audit_logs` table |
| Face identity verification | OpenCV LBPH enrolled owner model |
| Dashboard statistics | `/api/dashboard/statistics` |
| Charts | Existing Chart.js dashboard visualizations |
| Live events | `/api/events` and `/api/dashboard/events` |
| Access control | JWT-protected APIs + owner-only account |
| Account management | `/api/users` exposes only the authorized owner |
| Production-ready biometric claim | **Not claimed**; LBPH is an academic prototype |

## Scope decision

The supplied specification describes public registration, but the requested AI-ISF deployment is explicitly owner-only. Therefore public registration is intentionally disabled. The system seeds only the authorized owner account and does not expose additional demo users.

## Data integrity

The dashboard never seeds fake authentication events, fake alerts, or fake users. ML training uses synthetic internal data only to train the anomaly detector; those records are not displayed as security events.
