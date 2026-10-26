import os, io, base64, json, time, uuid, hashlib, sqlite3, logging, platform, socket, secrets, pickle
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from sklearn.ensemble import IsolationForest
import jwt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACE_DIR = DATA_DIR / "face"
MODEL_DIR = DATA_DIR / "ml"
DATA_DIR.mkdir(exist_ok=True); FACE_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "aisf.db"
FACE_MODEL_PATH = FACE_DIR / "jegan_sface_embeddings.npz"
FACE_ENROLLED_PATH = FACE_DIR / "enrolled.json"
FACE_THRESHOLD_PATH = FACE_DIR / "threshold.json"
YUNET_MODEL_PATH = FACE_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = FACE_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ML_MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SECRET_FILE = DATA_DIR / ".jwt_secret"
if os.environ.get("AISF_JWT_SECRET"):
    SECRET_KEY = os.environ["AISF_JWT_SECRET"]
else:
    if not SECRET_FILE.exists(): SECRET_FILE.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    SECRET_KEY = SECRET_FILE.read_text(encoding="utf-8").strip()
JWT_EXP_MINUTES = int(os.environ.get("AISF_JWT_EXP_MINUTES", "60"))
FACE_THRESHOLD = float(os.environ.get("AISF_FACE_THRESHOLD", "0.45"))
if FACE_THRESHOLD_PATH.exists():
    try: FACE_THRESHOLD = float(json.loads(FACE_THRESHOLD_PATH.read_text(encoding="utf-8")).get("threshold", FACE_THRESHOLD))
    except Exception: pass
AUTHORIZED_EMAIL = "jeganraj88888@gamil.com"
AUTHORIZED_NAME = "Jegan"
AUTHORIZED_ROLE = "Security Administrator"
OWNER_SCRYPT_SALT = bytes.fromhex("7a4d9b4f8a2c1d6e9f0b3c5d7e1a2f44")
OWNER_SCRYPT_HASH = "186e6048300e7d172064dd2f2bd95803b955411ca0b093cdda50db02cf9f2d13"

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AI-ISF")

CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Haar Cascade failed to load: {CASCADE_PATH}")
print(f"[OK] Haar Cascade loaded: {CASCADE_PATH}")


def password_hash(password):
    dk=hashlib.scrypt(password.encode(), salt=OWNER_SCRYPT_SALT, n=2**14, r=8, p=1, dklen=32)
    return dk.hex()

def check_password(password, stored_hash):
    return secrets.compare_digest(password_hash(password), stored_hash)


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db():
    con = db()
    con.executescript('''
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
      phone TEXT, password_hash TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ACTIVE'
    );
    CREATE TABLE IF NOT EXISTS authentication_events(
      id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, timestamp TEXT NOT NULL,
      ip_hash TEXT, ip_address TEXT, device_hash TEXT, location TEXT, host_name TEXT, endpoint TEXT, os_name TEXT, user_agent TEXT, success INTEGER NOT NULL,
      failed_attempts INTEGER NOT NULL DEFAULT 0, risk_score REAL NOT NULL, risk_level TEXT NOT NULL,
      action TEXT NOT NULL, event_type TEXT NOT NULL, result TEXT NOT NULL, detail TEXT,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS fraud_alerts(
      id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, event_id INTEGER, risk_score REAL,
      reason TEXT, status TEXT NOT NULL DEFAULT 'OPEN', created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(event_id) REFERENCES authentication_events(id)
    );
    CREATE TABLE IF NOT EXISTS sessions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, jti TEXT UNIQUE NOT NULL,
      device_hash TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS audit_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, event_type TEXT NOT NULL,
      description TEXT NOT NULL, timestamp TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE INDEX IF NOT EXISTS idx_auth_events_user_time ON authentication_events(user_id,timestamp);
    CREATE INDEX IF NOT EXISTS idx_alerts_status ON fraud_alerts(status);
    CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp);
    ''')
    # Non-destructive schema migration for real telemetry fields used by XDR filters.
    existing_cols={r[1] for r in con.execute("PRAGMA table_info(authentication_events)").fetchall()}
    for col, ddl in {
        "ip_address":"TEXT", "host_name":"TEXT", "endpoint":"TEXT",
        "os_name":"TEXT", "user_agent":"TEXT"
    }.items():
        if col not in existing_cols:
            con.execute(f"ALTER TABLE authentication_events ADD COLUMN {col} {ddl}")
    alert_cols={r[1] for r in con.execute("PRAGMA table_info(fraud_alerts)").fetchall()}
    for col, ddl in {"resolution_status":"TEXT", "closing_notes":"TEXT", "resolved_at":"TEXT"}.items():
        if col not in alert_cols:
            con.execute(f"ALTER TABLE fraud_alerts ADD COLUMN {col} {ddl}")
    con.execute("CREATE TABLE IF NOT EXISTS tenants(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL, tenant_key TEXT UNIQUE NOT NULL, status TEXT NOT NULL DEFAULT 'ACTIVE', notes TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))")
    con.execute("CREATE TABLE IF NOT EXISTS report_history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL, source TEXT NOT NULL, format TEXT NOT NULL, record_count INTEGER NOT NULL, query_text TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))")
    rh_cols={r[1] for r in con.execute("PRAGMA table_info(report_history)").fetchall()}
    if "query_text" not in rh_cols: con.execute("ALTER TABLE report_history ADD COLUMN query_text TEXT")
    row = con.execute("SELECT id FROM users WHERE email=?", (AUTHORIZED_EMAIL,)).fetchone()
    if not row:
        con.execute("INSERT INTO users(name,email,phone,password_hash,created_at,status) VALUES(?,?,?,?,?,?)",
                    (AUTHORIZED_NAME, AUTHORIZED_EMAIL, "", OWNER_SCRYPT_HASH, now(), "ACTIVE"))
    con.commit(); con.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def hash_identifier(value):
    return hashlib.sha256((value or "unknown").encode()).hexdigest()

def safe_int(value, default, minimum=None, maximum=None):
    try: n=int(value)
    except (TypeError, ValueError): n=default
    if minimum is not None: n=max(minimum,n)
    if maximum is not None: n=min(maximum,n)
    return n

def safe_float(value, default=None, minimum=None, maximum=None):
    try: n=float(value)
    except (TypeError, ValueError): return default
    if minimum is not None: n=max(minimum,n)
    if maximum is not None: n=min(maximum,n)
    return n


def public_user(row):
    return {"id": row["id"], "name": row["name"], "email": row["email"], "role": AUTHORIZED_ROLE, "status": row["status"]}


def get_owner():
    con = db(); row = con.execute("SELECT * FROM users WHERE email=?", (AUTHORIZED_EMAIL,)).fetchone(); con.close(); return row


def audit(user_id, event_type, description):
    con = db(); con.execute("INSERT INTO audit_logs(user_id,event_type,description,timestamp) VALUES(?,?,?,?)", (user_id,event_type,description,now())); con.commit(); con.close()


MODEL_FEATURES = ["failed_attempts", "login_hour", "daily_frequency", "new_device", "new_location", "time_since_previous_login_minutes", "successful_login_ratio"]

def _load_real_model():
    """Load only a real-telemetry model whose feature contract matches prediction."""
    meta_path = MODEL_DIR / "model_meta.json"
    if not ML_MODEL_PATH.exists() or not meta_path.exists(): return None
    try:
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source") != "real_authentication_events" or meta.get("features") != MODEL_FEATURES: return None
        with open(ML_MODEL_PATH,"rb") as f: model=pickle.load(f)
        if getattr(model,"n_features_in_",len(MODEL_FEATURES)) != len(MODEL_FEATURES): return None
        return model
    except Exception as exc:
        log.warning("Could not load fraud model: %s", exc); return None

def _event_feature_matrix(user_id):
    con=db(); rows=con.execute("SELECT timestamp,failed_attempts,device_hash,location,success FROM authentication_events WHERE user_id=? ORDER BY id",(user_id,)).fetchall(); con.close()
    if not rows: return np.empty((0,len(MODEL_FEATURES)),dtype=float)
    seen_devices=set(); seen_locations=set(); daily={}; X=[]; previous=None; successes=0; total_seen=0
    for r in rows:
        dt=datetime.fromisoformat(r["timestamp"]); day=dt.date().isoformat(); daily[day]=daily.get(day,0)+1
        dev=r["device_hash"] or ""; loc=r["location"] or ""
        new_dev=int(bool(dev) and dev not in seen_devices); new_loc=int(bool(loc) and loc not in seen_locations)
        seen_devices.add(dev); seen_locations.add(loc)
        delta=9999.0 if previous is None else max(0.0,(dt-previous).total_seconds()/60.0)
        successes += int(r["success"] or 0); total_seen += 1
        ratio=successes / total_seen
        X.append([float(r["failed_attempts"] or 0),float(dt.hour),float(daily[day]),new_dev,new_loc,delta,ratio])
        previous=dt
    return np.asarray(X,dtype=float)

def train_real_model_if_ready(user_id):
    global ML_MODEL
    con=db(); count=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=?",(user_id,)).fetchone()["c"]; con.close()
    if count < 20: return None
    X=_event_feature_matrix(user_id)
    if len(X)<20: return None
    model=IsolationForest(n_estimators=150, contamination="auto", random_state=42); model.fit(X)
    with open(ML_MODEL_PATH,"wb") as f: pickle.dump(model,f)
    (MODEL_DIR/"model_meta.json").write_text(json.dumps({"source":"real_authentication_events","user_id":user_id,"event_count":int(count),"trained_at":now(),"features":MODEL_FEATURES},indent=2),encoding="utf-8")
    ML_MODEL=model; return model

ML_MODEL = _load_real_model()

def risk_analysis(user_id, device_hash, location, failed_attempts, event_time=None):
    global ML_MODEL
    con = db(); rows = con.execute("SELECT device_hash,location,timestamp,success FROM authentication_events WHERE user_id=? ORDER BY id DESC LIMIT 100", (user_id,)).fetchall(); con.close()
    current = event_time or datetime.now(timezone.utc)
    previous = rows[0] if rows else None
    new_device = not any(r["device_hash"] == device_hash for r in rows)
    new_location = bool(location and not any((r["location"] or "") == location for r in rows))
    hour = current.hour
    recent_count = sum(1 for r in rows if r["timestamp"][:10] == current.date().isoformat())
    successes = sum(int(r["success"]) for r in rows)
    ratio = successes / len(rows) if rows else 1.0
    delta = 9999 if not previous else max(0, (current - datetime.fromisoformat(previous["timestamp"])).total_seconds()/60)
    features = np.array([[failed_attempts,hour,recent_count+1,int(new_device),int(new_location),delta,ratio]], dtype=float)
    if ML_MODEL is not None:
        try:
            meta=json.loads((MODEL_DIR/"model_meta.json").read_text(encoding="utf-8")); trained_count=int(meta.get("event_count",0))
            con=db(); total_events=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=?",(user_id,)).fetchone()["c"]; con.close()
            if meta.get("user_id") != user_id or trained_count < total_events: ML_MODEL=train_real_model_if_ready(user_id)
        except Exception:
            ML_MODEL=train_real_model_if_ready(user_id)
    else:
        ML_MODEL=train_real_model_if_ready(user_id)
    anomaly = bool(ML_MODEL is not None and int(ML_MODEL.predict(features)[0]) == -1)
    score = 0; reasons=[]
    if failed_attempts >= 5: score += 35; reasons.append("Multiple failed attempts")
    elif failed_attempts >= 3: score += 20; reasons.append("Repeated failed attempts")
    if new_device: score += 18; reasons.append("New device")
    if new_location: score += 18; reasons.append("New location")
    if hour < 6 or hour > 22: score += 12; reasons.append("Unusual login time")
    if recent_count >= 10: score += 12; reasons.append("High authentication frequency")
    if anomaly: score += 20; reasons.append("ML anomaly detected")
    score = min(100, score)
    if score >= 71: level,action="HIGH","BLOCK"
    elif score >= 31: level,action="MEDIUM","VERIFY"
    else: level,action="LOW","ALLOW"
    return {"risk_score": score, "risk_level": level, "reasons": reasons or ["Known authentication pattern"], "recommended_action": action,
            "features": {"login_hour":hour,"failed_attempts":failed_attempts,"login_frequency":recent_count+1,"new_device":new_device,"new_location":new_location,"time_since_previous_login":delta,"successful_login_ratio":round(ratio,3),"ml_anomaly":anomaly}}


def create_event(user_id, event_type, success, result, risk, action, ip_hash, device_hash, location, failed_attempts, detail, telemetry=None):
    telemetry=telemetry or {}
    con=db(); ts=now(); cur=con.execute("INSERT INTO authentication_events(user_id,timestamp,ip_hash,ip_address,device_hash,location,host_name,endpoint,os_name,user_agent,success,failed_attempts,risk_score,risk_level,action,event_type,result,detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (user_id,ts,ip_hash,telemetry.get("ip_address"),device_hash,location,telemetry.get("host_name"),telemetry.get("endpoint"),telemetry.get("os_name"),telemetry.get("user_agent"),int(success),failed_attempts,risk["risk_score"],risk["risk_level"],action,event_type,result,detail + (" | Risk reasons: " + "; ".join(risk.get("reasons",[])) if risk.get("reasons") else "")))
    event_id=cur.lastrowid
    if risk["risk_score"] >= 71:
        con.execute("INSERT INTO fraud_alerts(user_id,event_id,risk_score,reason,status,created_at) VALUES(?,?,?,?,?,?)",(user_id,event_id,risk["risk_score"],"; ".join(risk["reasons"]),"OPEN",ts))
    con.commit(); con.close(); audit(user_id,event_type,detail)
    return event_id,ts


def token_for(user_id, device_hash):
    jti=secrets.token_urlsafe(20); issued=datetime.now(timezone.utc); exp=issued+timedelta(minutes=JWT_EXP_MINUTES)
    con=db(); con.execute("INSERT INTO sessions(user_id,jti,device_hash,created_at,expires_at) VALUES(?,?,?,?,?)",(user_id,jti,device_hash,issued.isoformat(),exp.isoformat())); con.commit(); con.close()
    token=jwt.encode({"sub":user_id,"jti":jti,"iat":issued,"exp":exp}, SECRET_KEY, algorithm="HS256")
    return token, jti


def _session_user_by_jti(jti):
    if not jti: return None
    con=db(); row=con.execute("SELECT u.* FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.jti=? AND s.revoked=0 AND s.expires_at>?",(jti,now())).fetchone(); con.close()
    return row


def auth_user():
    # Accept the Authorization bearer first, then the JWT cookie, then an opaque
    # server-side session cookie. This makes the post-biometric browser handoff
    # resilient even when browser storage is unavailable.
    header=request.headers.get("Authorization","")
    candidates=[]
    if header.startswith("Bearer "): candidates.append(("bearer",header[7:].strip()))
    if request.cookies.get("aisf_access_token"): candidates.append(("jwt_cookie",request.cookies.get("aisf_access_token")))
    if request.cookies.get("aisf_session"): candidates.append(("session_cookie",request.cookies.get("aisf_session")))
    for kind,value in candidates:
        try:
            if kind == "session_cookie":
                row=_session_user_by_jti(value)
                if row is not None: return row
                continue
            payload=jwt.decode(value,SECRET_KEY,algorithms=["HS256"])
            row=_session_user_by_jti(payload.get("jti"))
            if row is not None: return row
        except Exception as exc:
            log.debug("Authentication candidate rejected (%s): %s", kind, exc)
    return None


def require_auth(fn):
    def wrapper(*args,**kwargs):
        user=auth_user()
        if not user: return jsonify({"success":False,"message":"Authentication required."}),401
        return fn(user,*args,**kwargs)
    wrapper.__name__=fn.__name__; return wrapper




def decode_image(data):
    try:
        encoded=data.split(",",1)[1] if data.startswith("data:") else data
        raw=base64.b64decode(encoded); arr=np.frombuffer(raw,np.uint8); return cv2.imdecode(arr,cv2.IMREAD_COLOR)
    except Exception: return None


def _download_model(url, path):
    if path.exists() and path.stat().st_size > 10000:
        return True
    try:
        import urllib.request
        tmp=path.with_suffix(path.suffix+".download")
        req=urllib.request.Request(url, headers={"User-Agent":"AI-ISF/13.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(tmp,"wb") as f:
            while True:
                chunk=r.read(1024*1024)
                if not chunk: break
                f.write(chunk)
        tmp.replace(path)
        return path.exists() and path.stat().st_size > 10000
    except Exception as exc:
        log.error("Face model download failed: %s", exc)
        try:
            tmp=path.with_suffix(path.suffix+".download")
            if tmp.exists(): tmp.unlink()
        except Exception: pass
        return False


def sface_available():
    return hasattr(cv2, "FaceRecognizerSF") and hasattr(cv2, "FaceDetectorYN")


def ensure_face_models():
    if not sface_available():
        return False
    y=_download_model(YUNET_URL, YUNET_MODEL_PATH)
    s=_download_model(SFACE_URL, SFACE_MODEL_PATH)
    return y and s


def load_sface_engine():
    global yunet_detector
    if not ensure_face_models(): return None, None
    try:
        recognizer=cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")
        detector=cv2.FaceDetectorYN.create(str(YUNET_MODEL_PATH), "", (320,320), 0.85, 0.30, 5000)
        return detector, recognizer
    except Exception as exc:
        log.error("SFace/YuNet initialization failed: %s", exc)
        return None, None


yunet_detector, sface_recognizer = load_sface_engine()


def detect_face_sface(img):
    if yunet_detector is None or sface_recognizer is None: return None, 0
    try:
        h,w=img.shape[:2]
        yunet_detector.setInputSize((w,h))
        _, faces=yunet_detector.detect(img)
        if faces is None or len(faces)!=1: return None, 0 if faces is None else len(faces)
        return faces[0], 1
    except Exception:
        return None, 0


def extract_embedding(img):
    face,count=detect_face_sface(img)
    if face is None: return None,count
    try:
        aligned=sface_recognizer.alignCrop(img, face)
        feature=sface_recognizer.feature(aligned)
        feature=np.asarray(feature,dtype=np.float32).reshape(1,-1)
        return feature,count
    except Exception:
        return None,count


def load_owner_embeddings():
    if not FACE_MODEL_PATH.exists(): return []
    try:
        data=np.load(FACE_MODEL_PATH)
        return [np.asarray(x,dtype=np.float32).reshape(1,-1) for x in data["embeddings"]]
    except Exception:
        return []

owner_embeddings=load_owner_embeddings()
preauth={}


@app.get("/")
def root(): return send_from_directory(str(BASE_DIR),"index.html")

@app.get("/dashboard")
def dashboard_route(): return send_from_directory(str(BASE_DIR),"dashboard.html")

@app.get("/<path:filename>")
def static_file(filename):
    if filename.startswith("api/"): return jsonify({"success":False,"message":"Not found"}),404
    return send_from_directory(str(BASE_DIR),filename)

@app.get("/api/health")
def health(): return jsonify({"status":"ok","service":"AI-ISF Backend","opencv_version":cv2.__version__,"cascade_loaded":not face_cascade.empty(),"ml_engine":"Isolation Forest available","model_trained":ML_MODEL is not None,"timestamp":now()})

@app.get("/api/system")
def system():
    try:
        import psutil; cpu=psutil.cpu_percent(.05); mem=psutil.virtual_memory().percent; disk=psutil.disk_usage(os.path.abspath(os.sep)).percent
    except Exception: cpu=mem=disk=None
    return jsonify({"success":True,"status":"online","service":"AI-ISF Backend","hostname":socket.gethostname(),"platform":platform.system(),"python_version":platform.python_version(),"opencv_version":cv2.__version__,"cascade_loaded":not face_cascade.empty(),"ml_engine":"Isolation Forest","cpu_percent":cpu,"memory_percent":mem,"disk_percent":disk,"timestamp":now()})

@app.get("/api/diagnostics")
@require_auth
def diagnostics(user):
    checks={"flask":True,"opencv":True,"haar_cascade":not face_cascade.empty(),"face_recognition":sface_available() and sface_recognizer is not None,"ml_model":ML_MODEL is not None,"database":DB_PATH.exists(),"event_log":True}
    return jsonify({"success":True,"status":"healthy" if all(checks.values()) else "degraded","checks":checks,"components":checks,"timestamp":now()})

@app.post("/api/auth/register")
def register_disabled():
    return jsonify({"success":False,"message":"Public registration is disabled in this single-owner AI-ISF deployment."}),403



def telemetry_from_request(endpoint=None):
    ua=request.headers.get("User-Agent","")
    ua_low=ua.lower()
    if "windows" in ua_low: os_name="Windows"
    elif "android" in ua_low: os_name="Android"
    elif "iphone" in ua_low or "ipad" in ua_low or "ios" in ua_low: os_name="iOS"
    elif "mac os" in ua_low or "macintosh" in ua_low: os_name="macOS"
    elif "linux" in ua_low: os_name="Linux"
    else: os_name="Unknown"
    return {
        "ip_address": request.headers.get("X-Forwarded-For", request.remote_addr or "Unknown").split(",")[0].strip(),
        "host_name": request.host,
        "endpoint": endpoint or request.path,
        "os_name": os_name,
        "user_agent": ua[:300],
    }


@app.post("/api/login")
def login():
    data=request.get_json(silent=True) or {}; email=(data.get("email") or "").strip().lower(); password=data.get("password","")
    device=hash_identifier(data.get("device_id") or request.headers.get("User-Agent","browser")); location=(data.get("location") or "Unknown")[:80]; raw_ip=(request.headers.get("X-Forwarded-For", request.remote_addr or "Unknown").split(",")[0].strip()); ip=hash_identifier(raw_ip); telemetry=telemetry_from_request("/api/login")
    con=db(); user=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone(); con.close()
    if not user or not check_password(password, user["password_hash"]):
        # count recent failed attempts
        con=db(); failed=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=? AND success=0 AND timestamp>=?",(user["id"] if user else None,(datetime.now(timezone.utc)-timedelta(minutes=15)).isoformat())).fetchone()["c"] if user else 1; con.close()
        if user:
            risk=risk_analysis(user["id"],device,location,failed+1); create_event(user["id"],"Authentication",False,"FAILED",risk,risk["recommended_action"],ip,device,location,failed+1,"Password authentication rejected",telemetry)
        return jsonify({"success":False,"message":"Invalid credentials."}),401
    if user["status"] != "ACTIVE": return jsonify({"success":False,"message":"Account is not active."}),403
    con=db(); failed=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=? AND success=0 AND timestamp>=?",(user["id"],(datetime.now(timezone.utc)-timedelta(minutes=15)).isoformat())).fetchone()["c"]; con.close()
    risk=risk_analysis(user["id"],device,location,failed)
    # Password authentication never grants access by itself. The enrolled face is always the final identity factor.
    # High-risk behavior is carried into the biometric step so a legitimate owner can complete step-up verification.
    pre=secrets.token_urlsafe(32); preauth[pre]={"user_id":user["id"],"device":device,"location":location,"ip":ip,"risk":risk,"expires":time.time()+180}
    create_event(user["id"],"Password Authentication",True,"PASSWORD_OK",risk,"VERIFY",ip,device,location,failed,"Password accepted; face identity verification required",telemetry)
    return jsonify({"success":True,"requires_face":True,"challenge":pre,"risk_score":risk["risk_score"],"risk_level":risk["risk_level"],"reasons":risk["reasons"],"user":public_user(user),"message":"Password accepted. Complete face identity verification."})


@app.post("/api/face/enroll")
def enroll_face():
    global owner_embeddings, FACE_THRESHOLD
    data=request.get_json(silent=True) or {}; email=(data.get("email") or "").strip().lower(); password=data.get("password",""); images=data.get("images") or []
    user=get_owner()
    if email!=AUTHORIZED_EMAIL or not check_password(password, user["password_hash"]):
        return jsonify({"success":False,"message":"Owner credentials are required for face enrollment."}),401
    if not sface_available() or yunet_detector is None or sface_recognizer is None:
        return jsonify({"success":False,"message":"AI face-recognition models are unavailable. Start the server with internet access once so YuNet and SFace can be downloaded."}),503
    if len(images)<8: return jsonify({"success":False,"message":"Capture at least 8 live face samples."}),400
    embeddings=[]
    for enc in images[:30]:
        img=decode_image(enc)
        if img is None: continue
        feat,count=extract_embedding(img)
        if feat is not None: embeddings.append(feat)
    if len(embeddings)<8:
        return jsonify({"success":False,"message":f"Only {len(embeddings)} usable face samples captured. Improve lighting and keep one face centered."}),400
    matrix=np.vstack(embeddings).astype(np.float32)
    # Normalize each embedding and store multiple owner templates. This is an identity
    # embedding system rather than a single-class classifier, so an unknown person is
    # not automatically assigned to the owner's class.
    norms=np.linalg.norm(matrix,axis=1,keepdims=True); matrix=matrix/np.maximum(norms,1e-8)
    np.savez_compressed(FACE_MODEL_PATH, embeddings=matrix)
    owner_embeddings=[row.reshape(1,-1) for row in matrix]
    # Calibrate from genuine owner-to-owner sample similarity. Keep a conservative floor.
    sims=[]
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            sims.append(float(np.dot(matrix[i],matrix[j])))
    if sims:
        calibrated=float(np.clip(np.percentile(sims,5)-0.03,0.40,0.55))
    else: calibrated=0.45
    FACE_THRESHOLD=calibrated
    FACE_THRESHOLD_PATH.write_text(json.dumps({"threshold":FACE_THRESHOLD,"calibrated_at":now(),"engine":"SFace-cosine"}),encoding="utf-8")
    FACE_ENROLLED_PATH.write_text(json.dumps({"email":AUTHORIZED_EMAIL,"samples":len(matrix),"enrolled_at":now(),"threshold":FACE_THRESHOLD,"engine":"OpenCV SFace + YuNet","version":13}),encoding="utf-8")
    audit(user["id"],"Face Enrollment",f"Owner SFace biometric template enrolled with {len(matrix)} embeddings; calibrated cosine threshold={FACE_THRESHOLD:.3f}")
    return jsonify({"success":True,"samples":len(matrix),"threshold":FACE_THRESHOLD,"engine":"SFace","message":"Your face identity has been enrolled using AI face embeddings and calibrated successfully."})


@app.get("/api/face/status")
def face_status():
    return jsonify({"success":True,"enrolled":FACE_ENROLLED_PATH.exists() and len(owner_embeddings)>0,"recognition_available":sface_available() and sface_recognizer is not None,"threshold":FACE_THRESHOLD,"engine":"SFace + YuNet"})


@app.post("/api/verify-face")
def verify_face():
    data=request.get_json(silent=True) or {}; challenge=data.get("challenge",""); images=data.get("images") or []
    if not images and data.get("image"): images=[data.get("image")]
    item=preauth.pop(challenge,None)
    if not item or item["expires"]<time.time(): return jsonify({"success":False,"message":"Face verification session expired. Sign in again."}),401
    con=db(); user=con.execute("SELECT * FROM users WHERE id=?",(item["user_id"],)).fetchone(); con.close()
    if len(owner_embeddings)==0: return jsonify({"success":False,"message":"Your face is not enrolled yet."}),409
    if len(images)<3: return jsonify({"success":False,"message":"Capture several live frames for reliable identity verification."}),400
    observations=[]; valid=0
    template_matrix=np.vstack(owner_embeddings).astype(np.float32)
    for enc in images[:9]:
        img=decode_image(enc)
        if img is None: continue
        feat,count=extract_embedding(img)
        if feat is None: continue
        valid += 1
        feat=feat/np.maximum(np.linalg.norm(feat),1e-8)
        scores=np.dot(template_matrix,feat.reshape(-1))
        best=float(np.max(scores))
        observations.append(best)
    if not observations:
        risk={"risk_score":90,"risk_level":"HIGH","recommended_action":"BLOCK","reasons":["No clear face detected in verification frames"]}
        eid,ts=create_event(user["id"],"Face Identity Verification",False,"FAILED",risk,"BLOCK",item["ip"],item["device"],item["location"],0,"No usable face frames",telemetry_from_request("/api/verify-face"))
        return jsonify({"success":True,"result":"FAILED","face_detected":False,"face_count":0,"event_id":eid,"message":"No clear face was detected. Improve lighting and center your face."})
    threshold=float(FACE_THRESHOLD)
    matches=[score for score in observations if score>=threshold]
    match_ratio=len(matches)/len(observations)
    # Require multiple independent frames and a strong median score. This is a true
    # unknown-rejection comparison; an unknown face is never assigned the owner's label.
    top_scores=sorted(observations,reverse=True)
    evidence=top_scores[:min(3,len(top_scores))]
    evidence_score=float(np.median(evidence))
    required=max(3,int(np.ceil(len(observations)*0.50)))
    matched=len(matches)>=required and evidence_score>=threshold
    score=max(0,min(100,evidence_score*100))
    base=item["risk"]
    final_score=min(100,max(0,base["risk_score"] if matched else base["risk_score"]+55))
    level="HIGH" if final_score>=71 else "MEDIUM" if final_score>=31 else "LOW"
    action="ALLOW" if matched and final_score<71 else "BLOCK"
    result="PASSED" if action=="ALLOW" else "FAILED"
    reasons=[] if matched else ["Face does not match the enrolled owner identity"]
    if matched and final_score>=71: reasons=["Identity matched but authentication behavior is high risk"]
    detail=f"identity_match={matched}, valid_frames={valid}, matched_frames={len(matches)}, match_ratio={match_ratio:.2f}, best_similarity={max(observations):.3f}, evidence_similarity={evidence_score:.3f}, threshold={threshold:.3f}, required_frames={required}, engine=SFace"
    eid,ts=create_event(user["id"],"Face Identity Verification",result=="PASSED",result,{"risk_score":final_score,"risk_level":level,"recommended_action":action,"reasons":reasons},action,item["ip"],item["device"],item["location"],0,detail,telemetry_from_request("/api/verify-face"))
    resp={"success":True,"result":result,"face_detected":True,"face_count":1,"valid_frames":valid,"matched_frames":len(matches),"match_ratio":round(match_ratio,2),"match_score":round(score,1),"similarity":round(evidence_score,3),"threshold":threshold,"risk_score":final_score,"risk_level":level,"event_id":eid,"timestamp":ts,"message":(f"Face identity matched using SFace. {len(matches)}/{valid} frames accepted (similarity {evidence_score:.3f}, threshold {threshold:.3f}). Access granted." if result=="PASSED" else f"Identity verification failed. {len(matches)}/{valid} frames matched (similarity {evidence_score:.3f}, threshold {threshold:.3f}). This face is not sufficiently similar to the enrolled owner.")}
    if result=="PASSED":
        access_token,jti=token_for(user["id"],item["device"]); resp.update({"access_token":access_token,"user":public_user(user),"redirect":"/dashboard"})
        response=make_response(jsonify(resp)); response.set_cookie("aisf_access_token",access_token,max_age=JWT_EXP_MINUTES*60,httponly=True,samesite="Lax",secure=False,path="/"); response.set_cookie("aisf_session",jti,max_age=JWT_EXP_MINUTES*60,httponly=True,samesite="Lax",secure=False,path="/"); return response
    return jsonify(resp)

@app.get("/api/auth/me")
@require_auth
def me(user): return jsonify({"success":True,"user":public_user(user)})

@app.post("/api/auth/logout")
@require_auth
def logout(user):
    h=request.headers.get("Authorization",""); token=h[7:].strip() if h.startswith("Bearer ") else request.cookies.get("aisf_access_token","")
    jti=request.cookies.get("aisf_session","")
    if token:
        try: jti=jwt.decode(token,SECRET_KEY,algorithms=["HS256"],options={"verify_exp":False}).get("jti",jti)
        except Exception: pass
    con=db(); con.execute("UPDATE sessions SET revoked=1 WHERE jti=?",(jti,)); con.commit(); con.close(); audit(user["id"],"Logout","Session revoked")
    response=make_response(jsonify({"success":True}))
    response.delete_cookie("aisf_access_token",path="/")
    response.delete_cookie("aisf_session",path="/")
    return response

@app.post("/api/risk/analyze")
@require_auth
def risk_api(user):
    data=request.get_json(silent=True) or {}; device=hash_identifier(data.get("device_id") or request.headers.get("User-Agent","browser")); loc=data.get("location") or "Unknown"; failed=safe_int(data.get("failed_attempts",0),0,0,1000); return jsonify(risk_analysis(user["id"],device,loc,failed))


@app.get("/api/model/status")
@require_auth
def model_status(user):
    meta_path=MODEL_DIR/"model_meta.json"
    meta={}
    if meta_path.exists():
        try: meta=json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception: meta={}
    con=db(); count=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=?",(user["id"],)).fetchone()["c"]; con.close()
    return jsonify({"success":True,"trained":ML_MODEL is not None,"status":"TRAINED" if ML_MODEL is not None else "NOT_TRAINED","real_event_count":count,"minimum_events":20,"metadata":meta})
@app.get("/api/dashboard/statistics")
@require_auth
def dashboard_stats(user):
    con=db(); ev=con.execute("SELECT COUNT(*) total, SUM(success) successes, SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) failures, SUM(CASE WHEN risk_score>=71 THEN 1 ELSE 0 END) high, SUM(CASE WHEN risk_score BETWEEN 31 AND 70 THEN 1 ELSE 0 END) medium, AVG(risk_score) avg_risk FROM authentication_events WHERE user_id=?",(user["id"],)).fetchone(); alerts=con.execute("SELECT COUNT(*) c FROM fraud_alerts WHERE user_id=? AND status='OPEN'",(user["id"],)).fetchone()["c"]; users=con.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]; con.close();
    return jsonify({"success":True,"total_users":users,"total_attempts":ev["total"] or 0,"successful_logins":ev["successes"] or 0,"failed_logins":ev["failures"] or 0,"high_risk_events":ev["high"] or 0,"medium_risk_events":ev["medium"] or 0,"open_alerts":alerts,"average_risk_score":round(ev["avg_risk"] or 0,1)})

@app.get("/api/events")
@require_auth
def events(user):
    limit=min(int(request.args.get("limit",200)),500)
    con=db(); rows=con.execute("""SELECT e.id,e.timestamp,e.event_type,e.result,e.risk_score,e.risk_level,e.action,e.location,e.detail,
        e.ip_address,e.host_name,e.endpoint,e.os_name,e.device_hash,
        a.id AS alert_id,a.status AS case_status,a.resolution_status,a.closing_notes,a.resolved_at
        FROM authentication_events e LEFT JOIN fraud_alerts a ON a.event_id=e.id AND a.user_id=e.user_id
        WHERE e.user_id=? ORDER BY e.id DESC LIMIT ?""",(user["id"],limit)).fetchall(); con.close()
    out=[]
    for r in rows:
        out.append({"id":f"EVT-{r['id']:06d}","timestamp":r["timestamp"],"event_type":r["event_type"],"result":r["result"],"risk":r["risk_score"],"risk_level":r["risk_level"],"action":r["action"],"location":r["location"],"ip":r["ip_address"] or "Unknown","host":r["host_name"] or "Unknown","endpoint":r["endpoint"] or "Unknown","os":r["os_name"] or "Unknown","device":r["device_hash"] or "Unknown","user":user["email"],"detail":r["detail"],"alert_id":r["alert_id"],"case_status":r["case_status"],"resolution_status":r["resolution_status"],"closing_notes":r["closing_notes"],"resolved_at":r["resolved_at"]})
    return jsonify({"success":True,"events":out,"total":len(out)})

@app.get("/api/dashboard/events")
@require_auth
def dashboard_events(user): return events(user)

@app.get("/api/dashboard/alerts")
@require_auth
def dashboard_alerts(user):
    con=db(); rows=con.execute("SELECT id,event_id,risk_score,reason,status,created_at FROM fraud_alerts WHERE user_id=? ORDER BY id DESC LIMIT 200",(user["id"],)).fetchall(); con.close(); return jsonify({"success":True,"alerts":[dict(r) for r in rows]})

@app.get("/api/alerts")
@require_auth
def alerts(user): return dashboard_alerts(user)

@app.post('/api/events/<int:event_id>/case')
@require_auth
def create_event_case(user,event_id):
    con=db()
    ev=con.execute('SELECT id,user_id,risk_score,risk_level,event_type,detail FROM authentication_events WHERE id=? AND user_id=?',(event_id,user['id'])).fetchone()
    if not ev:
        con.close(); return jsonify({'success':False,'message':'Detection not found.'}),404
    existing=con.execute('SELECT id,status FROM fraud_alerts WHERE event_id=? AND user_id=? ORDER BY id DESC LIMIT 1',(event_id,user['id'])).fetchone()
    if existing:
        con.close(); return jsonify({'success':True,'alert_id':existing['id'],'status':existing['status'],'message':'Case already exists.'})
    reason=f"Case manually opened from detection: {ev['event_type']}"
    cur=con.execute("INSERT INTO fraud_alerts(user_id,event_id,risk_score,reason,status,created_at) VALUES(?,?,?,?,?,?)",(user['id'],event_id,ev['risk_score'],reason,'OPEN',datetime.now(timezone.utc).isoformat()))
    con.commit(); alert_id=cur.lastrowid; con.close()
    audit(user['id'],'Case Created',f'Case {alert_id} created from detection EVT-{event_id:06d}')
    return jsonify({'success':True,'alert_id':alert_id,'status':'OPEN'})

@app.post("/api/admin/alerts/<int:alert_id>/resolve")
@require_auth
def resolve_alert(user,alert_id):
    data=request.get_json(silent=True) or {}; resolution=(data.get("status") or "TRUE_POSITIVE_BENIGN").upper(); notes=(data.get("closing_notes") or "").strip()[:2000]
    if resolution not in {"TRUE_POSITIVE_BENIGN","TRUE_POSITIVE_MALICIOUS","FALSE_POSITIVE","NOT_ACTIONABLE"}: return jsonify({"success":False,"message":"Invalid resolution status."}),400
    if not notes: return jsonify({"success":False,"message":"Closing notes are required."}),400
    con=db(); cur=con.execute("UPDATE fraud_alerts SET status='RESOLVED',resolution_status=?,closing_notes=?,resolved_at=? WHERE id=? AND user_id=?",(resolution,notes,now(),alert_id,user["id"])); con.commit(); con.close(); audit(user["id"],"Detection Resolved",f"Alert {alert_id} resolved as {resolution}. Closing notes: {notes}"); return jsonify({"success":cur.rowcount>0,"status":"RESOLVED","resolution_status":resolution,"closing_notes":notes})

@app.post("/api/events/<int:event_id>/triage")
@require_auth
def triage_event(user,event_id):
    data=request.get_json(silent=True) or {}
    resolution=(data.get("status") or "").strip().upper()
    notes=(data.get("closing_notes") or "").strip()[:2000]
    allowed={"OPEN","TRUE_POSITIVE_BENIGN","TRUE_POSITIVE_MALICIOUS","FALSE_POSITIVE","NOT_ACTIONABLE"}
    if resolution not in allowed:
        return jsonify({"success":False,"message":"Choose a valid resolution status."}),400
    if resolution!="OPEN" and not notes:
        return jsonify({"success":False,"message":"Closing notes are required when resolving a detection."}),400
    con=db(); ev=con.execute("SELECT id,event_type,risk_score FROM authentication_events WHERE id=? AND user_id=?",(event_id,user["id"])).fetchone()
    if not ev: con.close(); return jsonify({"success":False,"message":"Detection not found."}),404
    alert=con.execute("SELECT id FROM fraud_alerts WHERE event_id=? AND user_id=? ORDER BY id DESC LIMIT 1",(event_id,user["id"])).fetchone()
    if not alert:
        cur=con.execute("INSERT INTO fraud_alerts(user_id,event_id,risk_score,reason,status,resolution_status,closing_notes,resolved_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(user["id"],event_id,ev["risk_score"],f"Case opened from detection: {ev['event_type']}","OPEN",resolution,notes,now() if resolution!="OPEN" else None,now()))
        alert_id=cur.lastrowid
    else:
        alert_id=alert["id"]
        status="OPEN" if resolution=="OPEN" else "RESOLVED"
        con.execute("UPDATE fraud_alerts SET status=?,resolution_status=?,closing_notes=?,resolved_at=? WHERE id=? AND user_id=?",(status,resolution,notes,now() if resolution!="OPEN" else None,alert_id,user["id"]))
    con.commit(); con.close()
    audit(user["id"],"Detection Triaged",f"Detection EVT-{event_id:06d} marked {resolution}. Closing notes: {notes or 'Re-opened'}")
    return jsonify({"success":True,"alert_id":alert_id,"status":"OPEN" if resolution=="OPEN" else "RESOLVED","resolution_status":resolution,"closing_notes":notes})

@app.get("/api/cases")
@require_auth
def cases_api(user):
    con=db(); rows=con.execute("SELECT a.id,a.event_id,a.risk_score,a.reason,a.status,a.resolution_status,a.closing_notes,a.created_at,a.resolved_at,e.event_type,e.result,e.ip_address,e.host_name,e.endpoint,e.os_name FROM fraud_alerts a LEFT JOIN authentication_events e ON e.id=a.event_id WHERE a.user_id=? ORDER BY a.id DESC LIMIT 500",(user["id"],)).fetchall(); con.close(); return jsonify({"success":True,"cases":[dict(r) for r in rows]})

@app.get("/api/users")
@require_auth
def users(user): return jsonify({"success":True,"users":[public_user(user)]})

@app.get("/api/users/<int:user_id>")
@require_auth
def user_detail(user,user_id):
    if user_id!=user["id"]: return jsonify({"success":False,"message":"Access denied."}),403
    return jsonify({"success":True,"user":public_user(user)})

@app.post("/api/audit/page")
@require_auth
def audit_page(user):
    data=request.get_json(silent=True) or {}; page=(data.get("page") or "unknown")[:120]
    audit(user["id"],"Page Viewed",f"Opened {page}")
    return jsonify({"success":True})

@app.get("/api/audit-logs")
@require_auth
def audit_logs(user):
    limit=safe_int(request.args.get("limit",200),200,1,500); con=db(); rows=con.execute("SELECT id,event_type,description,timestamp FROM audit_logs WHERE user_id=? ORDER BY id DESC LIMIT ?",(user["id"],limit)).fetchall(); con.close(); return jsonify({"success":True,"logs":[dict(r) for r in rows]})

@app.get("/api/search")
@require_auth
def search_api(user):
    q=(request.args.get("q") or "").strip(); limit=safe_int(request.args.get("limit",500),500,1,1000)
    con=db(); like=f"%{q}%"
    rows=con.execute("""SELECT e.id,e.timestamp,e.event_type,e.result,e.risk_score,e.risk_level,e.detail,e.ip_address,e.host_name,e.endpoint,e.os_name,e.location,a.id alert_id,a.status case_status
      FROM authentication_events e LEFT JOIN fraud_alerts a ON a.event_id=e.id AND a.user_id=e.user_id
      WHERE e.user_id=? AND (?='' OR e.event_type LIKE ? OR e.result LIKE ? OR e.detail LIKE ? OR e.ip_address LIKE ? OR e.host_name LIKE ? OR e.endpoint LIKE ? OR e.os_name LIKE ? OR e.location LIKE ?) ORDER BY e.id DESC LIMIT ?""",(user["id"],q,like,like,like,like,like,like,like,like,limit)).fetchall()
    out=[{"source":"Detection","id":f"EVT-{r['id']:06d}","timestamp":r["timestamp"],"title":r["event_type"],"status":r["case_status"] or r["result"],"risk":r["risk_score"],"detail":r["detail"],"ip":r["ip_address"],"host":r["host_name"],"endpoint":r["endpoint"],"os":r["os_name"],"location":r["location"],"alert_id":r["alert_id"]} for r in rows]
    audit(user["id"],"Advanced Search",f"Searched evidence for: {q or '[all events]'}; returned {len(out)} records")
    con.close(); return jsonify({"success":True,"results":out,"total":len(out)})

@app.get("/api/search/advanced")
@require_auth
def advanced_search(user):
    q=(request.args.get("q") or "").strip(); preset=(request.args.get("preset") or "").strip(); ip=(request.args.get("ip") or "").strip(); host=(request.args.get("host") or "").strip(); endpoint=(request.args.get("endpoint") or "").strip(); os_name=(request.args.get("os") or "").strip(); event_type=(request.args.get("event_type") or "").strip(); risk_min=safe_float(request.args.get("risk_min"),None,0,100); since_hours=safe_float(request.args.get("since_hours"),None,0,8760); limit=safe_int(request.args.get("limit",1000),1000,1,5000)
    clauses=["e.user_id=?"]; args=[user["id"]]
    if q: like=f"%{q}%"; clauses.append("(e.event_type LIKE ? OR e.result LIKE ? OR e.detail LIKE ? OR e.ip_address LIKE ? OR e.host_name LIKE ? OR e.endpoint LIKE ? OR e.os_name LIKE ? OR e.location LIKE ?)"); args += [like]*8
    if ip: clauses.append("e.ip_address LIKE ?"); args.append(f"%{ip}%")
    if host: clauses.append("e.host_name LIKE ?"); args.append(f"%{host}%")
    if endpoint: clauses.append("e.endpoint LIKE ?"); args.append(f"%{endpoint}%")
    if os_name: clauses.append("e.os_name=?"); args.append(os_name)
    if event_type: clauses.append("e.event_type=?"); args.append(event_type)
    if risk_min: clauses.append("e.risk_score>=?"); args.append(float(risk_min))
    if since_hours: clauses.append("e.timestamp>=?"); args.append((datetime.now(timezone.utc)-timedelta(hours=float(since_hours))).isoformat())
    if preset=="failed_24h": clauses += ["e.success=0","e.timestamp>=?"]; args.append((datetime.now(timezone.utc)-timedelta(hours=24)).isoformat())
    elif preset=="high_risk": clauses.append("e.risk_score>=71")
    elif preset=="face_failures": clauses += ["e.event_type='Face Identity Verification'","e.success=0"]
    elif preset=="password_failures": clauses += ["e.event_type='Authentication'","e.success=0"]
    elif preset=="new_device": clauses.append("e.detail LIKE '%New device%'")
    where=" AND ".join(clauses)
    con=db(); rows=con.execute(f"SELECT e.id,e.timestamp,e.event_type,e.result,e.risk_score,e.risk_level,e.ip_address,e.host_name,e.endpoint,e.os_name,e.location,e.detail FROM authentication_events e WHERE {where} ORDER BY e.id DESC LIMIT ?",args+[limit]).fetchall(); con.close()
    out=[dict(r) for r in rows]
    audit(user["id"],"Advanced Search",f"Advanced query preset={preset or 'custom'} returned {len(out)} records")
    return jsonify({"success":True,"results":out,"total":len(out),"query":{"preset":preset,"q":q,"ip":ip,"host":host,"endpoint":endpoint,"os":os_name,"event_type":event_type,"risk_min":risk_min,"since_hours":since_hours}})

@app.get("/api/search/advanced/export")
@require_auth
def advanced_search_export(user):
    from flask import Response
    # Reuse the same filter construction without making an HTTP request.
    q=(request.args.get("q") or "").strip(); preset=(request.args.get("preset") or "").strip(); ip=(request.args.get("ip") or "").strip(); host=(request.args.get("host") or "").strip(); endpoint=(request.args.get("endpoint") or "").strip(); os_name=(request.args.get("os") or "").strip(); event_type=(request.args.get("event_type") or "").strip(); risk_min=safe_float(request.args.get("risk_min"),None,0,100); since_hours=safe_float(request.args.get("since_hours"),None,0,8760); fmt=request.args.get("format","csv").lower(); limit=safe_int(request.args.get("limit",5000),5000,1,5000)
    clauses=["user_id=?"]; args=[user["id"]]
    if q: like=f"%{q}%"; clauses.append("(event_type LIKE ? OR result LIKE ? OR detail LIKE ? OR ip_address LIKE ? OR host_name LIKE ? OR endpoint LIKE ? OR os_name LIKE ? OR location LIKE ?)"); args += [like]*8
    if ip: clauses.append("ip_address LIKE ?"); args.append(f"%{ip}%")
    if host: clauses.append("host_name LIKE ?"); args.append(f"%{host}%")
    if endpoint: clauses.append("endpoint LIKE ?"); args.append(f"%{endpoint}%")
    if os_name: clauses.append("os_name=?"); args.append(os_name)
    if event_type: clauses.append("event_type=?"); args.append(event_type)
    if risk_min: clauses.append("risk_score>=?"); args.append(float(risk_min))
    if since_hours: clauses.append("timestamp>=?"); args.append((datetime.now(timezone.utc)-timedelta(hours=float(since_hours))).isoformat())
    if preset=="failed_24h": clauses += ["success=0","timestamp>=?"]; args.append((datetime.now(timezone.utc)-timedelta(hours=24)).isoformat())
    elif preset=="high_risk": clauses.append("risk_score>=71")
    elif preset=="face_failures": clauses += ["event_type='Face Identity Verification'","success=0"]
    elif preset=="password_failures": clauses += ["event_type='Authentication'","success=0"]
    elif preset=="new_device": clauses.append("detail LIKE '%New device%'")
    con=db(); rows=con.execute(f"SELECT id,timestamp,event_type,result,risk_score,risk_level,ip_address,host_name,endpoint,os_name,location,detail FROM authentication_events WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",args+[limit]).fetchall(); con.close()
    if fmt=="json": payload=json.dumps([dict(r) for r in rows],indent=2); mime="application/json"; ext="json"
    else:
        out=io.StringIO(); import csv; w=csv.writer(out); w.writerow(rows[0].keys() if rows else ["id","timestamp","event_type","result","risk_score","risk_level","ip_address","host_name","endpoint","os_name","location","detail"]); [w.writerow(list(r)) for r in rows]; payload=out.getvalue(); mime="text/csv"; ext="csv"
    con=db(); cur=con.execute("INSERT INTO report_history(user_id,name,source,format,record_count,query_text,created_at) VALUES(?,?,?,?,?,?,?)",(user["id"],f"Advanced Search {datetime.now().strftime('%Y%m%d-%H%M%S')}","Advanced Search",ext,len(rows),json.dumps({"q":q,"preset":preset,"ip":ip,"host":host,"endpoint":endpoint,"os":os_name,"event_type":event_type,"risk_min":risk_min,"since_hours":since_hours}),now())); con.commit(); rid=cur.lastrowid; con.close(); audit(user["id"],"Data Export",f"Advanced Search report {rid} exported: {len(rows)} records")
    resp=make_response(payload); resp.headers["Content-Type"]=mime; resp.headers["Content-Disposition"]=f'attachment; filename="ai-isf-advanced-search-{rid}.{ext}"'; return resp

@app.get("/api/search/export")
@require_auth
def search_export(user):
    q=(request.args.get("q") or "").strip(); fmt=request.args.get("format","csv").lower(); limit=min(int(request.args.get("limit",1000)),5000)
    d=search_api.__wrapped__(user) if False else None
    con=db(); like=f"%{q}%"; rows=con.execute("SELECT id,timestamp,event_type,result,risk_score,risk_level,ip_address,host_name,endpoint,os_name,location,detail FROM authentication_events WHERE user_id=? AND (?='' OR event_type LIKE ? OR result LIKE ? OR detail LIKE ? OR ip_address LIKE ? OR host_name LIKE ? OR endpoint LIKE ? OR os_name LIKE ? OR location LIKE ?) ORDER BY id DESC LIMIT ?",(user["id"],q,like,like,like,like,like,like,like,limit)).fetchall(); con.close()
    import csv
    if fmt=="json": payload=json.dumps([dict(r) for r in rows],indent=2); mimetype="application/json"; ext="json"
    else:
        out=io.StringIO(); w=csv.writer(out); w.writerow(rows[0].keys() if rows else ["id","timestamp","event_type","result","risk_score","risk_level","ip_address","host_name","endpoint","os_name","location","detail"]); [w.writerow(list(r)) for r in rows]; payload=out.getvalue(); mimetype="text/csv"; ext="csv"
    con=db(); cur=con.execute("INSERT INTO report_history(user_id,name,source,format,record_count,query_text,created_at) VALUES(?,?,?,?,?,?,?)",(user["id"],f"Search {datetime.now().strftime('%Y%m%d-%H%M%S')}","Search",ext,len(rows),json.dumps({"q":q}),now())); con.commit(); report_id=cur.lastrowid; con.close(); audit(user["id"],"Data Export",f"Search export {report_id}: {len(rows)} records ({ext})")
    resp=make_response(payload); resp.headers["Content-Type"]=mimetype; resp.headers["Content-Disposition"]=f'attachment; filename="ai-isf-search-{report_id}.{ext}"'; return resp

@app.get("/api/download/events.csv")
@require_auth
def download_events(user):
    con=db(); rows=con.execute("SELECT id,timestamp,event_type,result,risk_score,risk_level,ip_address,host_name,endpoint,os_name,location,detail FROM authentication_events WHERE user_id=? ORDER BY id DESC",(user["id"],)).fetchall(); con.close(); out=io.StringIO(); import csv; w=csv.writer(out); w.writerow(rows[0].keys() if rows else ["id","timestamp","event_type","result","risk_score","risk_level","ip_address","host_name","endpoint","os_name","location","detail"]); [w.writerow(list(r)) for r in rows]; audit(user["id"],"Data Export",f"Authentication events CSV exported: {len(rows)} records"); resp=make_response(out.getvalue()); resp.headers["Content-Type"]="text/csv"; resp.headers["Content-Disposition"]='attachment; filename="ai-isf-authentication-events.csv"'; return resp

@app.get("/api/download/cases.csv")
@require_auth
def download_cases(user):
    con=db(); rows=con.execute("SELECT a.id,a.event_id,a.risk_score,a.status,a.resolution_status,a.closing_notes,a.created_at,a.resolved_at,e.event_type,e.result,e.ip_address,e.host_name,e.endpoint,e.os_name FROM fraud_alerts a LEFT JOIN authentication_events e ON e.id=a.event_id WHERE a.user_id=? ORDER BY a.id DESC",(user["id"],)).fetchall(); con.close(); out=io.StringIO(); import csv; w=csv.writer(out); w.writerow(rows[0].keys() if rows else ["id","event_id","risk_score","status","resolution_status","closing_notes","created_at","resolved_at","event_type","result","ip_address","host_name","endpoint","os_name"]); [w.writerow(list(r)) for r in rows]; audit(user["id"],"Data Export",f"Cases CSV exported: {len(rows)} records"); resp=make_response(out.getvalue()); resp.headers["Content-Type"]="text/csv"; resp.headers["Content-Disposition"]='attachment; filename="ai-isf-cases.csv"'; return resp

@app.get("/api/download/agents.csv")
@require_auth
def download_agents(user):
    con=db(); rows=con.execute("SELECT device_hash,host_name,os_name,COUNT(*) events,SUM(success) successes,SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) failures,MAX(timestamp) last_seen FROM authentication_events WHERE user_id=? GROUP BY device_hash,host_name,os_name ORDER BY last_seen DESC",(user["id"],)).fetchall(); con.close(); out=io.StringIO(); import csv; w=csv.writer(out); w.writerow(["device_hash","host_name","os_name","events","successes","failures","last_seen"]); [w.writerow(list(r)) for r in rows]; audit(user["id"],"Data Export",f"Endpoint agents CSV exported: {len(rows)} records"); resp=make_response(out.getvalue()); resp.headers["Content-Type"]="text/csv"; resp.headers["Content-Disposition"]='attachment; filename="ai-isf-endpoint-agents.csv"'; return resp

@app.get("/api/download/audit.csv")
@require_auth
def download_audit(user):
    con=db(); rows=con.execute("SELECT id,timestamp,event_type,description FROM audit_logs WHERE user_id=? ORDER BY id DESC",(user["id"],)).fetchall(); con.close(); out=io.StringIO(); import csv; w=csv.writer(out); w.writerow(["id","timestamp","event_type","description"]); [w.writerow(list(r)) for r in rows]; audit(user["id"],"Data Export",f"Audit log CSV exported: {len(rows)} records"); resp=make_response(out.getvalue()); resp.headers["Content-Type"]="text/csv"; resp.headers["Content-Disposition"]='attachment; filename="ai-isf-audit-logs.csv"'; return resp

@app.get("/api/reports/history")
@require_auth
def report_history(user):
    con=db(); rows=con.execute("SELECT id,name,source,format,record_count,query_text,created_at FROM report_history WHERE user_id=? ORDER BY id DESC LIMIT 200",(user["id"],)).fetchall(); con.close(); return jsonify({"success":True,"reports":[dict(r) for r in rows]})

@app.post("/api/reports/generate")
@require_auth
def generate_report(user):
    data=request.get_json(silent=True) or {}; source=data.get("source","events"); fmt=data.get("format","csv"); q=data.get("query","")
    if source=="events":
        url="/api/download/events.csv"; con=db(); count=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=?",(user["id"],)).fetchone()["c"]; con.close()
    elif source=="search": url="/api/search/export?format="+fmt+"&q="+__import__('urllib.parse').parse.quote(q); count=0
    elif source=="audit":
        url="/api/audit-logs?limit=1000"; con=db(); count=con.execute("SELECT COUNT(*) c FROM audit_logs WHERE user_id=?",(user["id"],)).fetchone()["c"]; con.close()
    else: return jsonify({"success":False,"message":"Unsupported report source."}),400
    con=db(); cur=con.execute("INSERT INTO report_history(user_id,name,source,format,record_count,query_text,created_at) VALUES(?,?,?,?,?,?,?)",(user["id"],f"{source.title()} Report {datetime.now().strftime('%Y%m%d-%H%M%S')}",source,fmt,count,json.dumps({"query":q}),now())); con.commit(); rid=cur.lastrowid; con.close(); audit(user["id"],"Report Created",f"Report {rid} created from {source}"); return jsonify({"success":True,"report_id":rid,"download_url":url,"record_count":count})

@app.get("/api/agents")
@require_auth
def agents(user):
    con=db(); rows=con.execute("SELECT device_hash,host_name,os_name,COUNT(*) events,SUM(success) successes,SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) failures,MAX(timestamp) last_seen FROM authentication_events WHERE user_id=? GROUP BY device_hash,host_name,os_name ORDER BY last_seen DESC",(user["id"],)).fetchall(); con.close(); out=[]
    for i,r in enumerate(rows,1): out.append({"id":f"AGENT-{i:04d}","device":r["host_name"] or "Unknown host","fingerprint":r["device_hash"] or "Unknown","os":r["os_name"] or "Unknown","events":r["events"],"successes":r["successes"] or 0,"failures":r["failures"] or 0,"last_seen":r["last_seen"],"status":"ACTIVE" if (datetime.now(timezone.utc)-datetime.fromisoformat(r["last_seen"])).total_seconds()<86400 else "STALE"})
    return jsonify({"success":True,"agents":out,"total":len(out)})

@app.get("/api/threat-research")
@require_auth
def threat_research(user):
    con=db(); rows=con.execute("SELECT event_type,risk_level,COUNT(*) events,MAX(timestamp) last_seen,detail FROM authentication_events WHERE user_id=? AND risk_score>=31 GROUP BY event_type,risk_level ORDER BY last_seen DESC",(user["id"],)).fetchall(); con.close(); findings=[]
    for r in rows:
        findings.append({"type":r["event_type"],"risk_level":r["risk_level"],"evidence":r["detail"] or "Observed risk telemetry","events":r["events"],"last_seen":r["last_seen"]})
    return jsonify({"success":True,"findings":findings,"source":"AI-ISF observed telemetry"})

@app.get("/api/threat-research/kev")
@require_auth
def threat_kev(user):
    import requests
    url="https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    try:
        r=requests.get(url,timeout=12); data=r.json(); audit(user["id"],"CTI Feed Access","CISA Known Exploited Vulnerabilities catalog queried"); return jsonify({"success":r.ok,"source":"CISA KEV","vulnerabilities":data.get("vulnerabilities",[])})
    except Exception as e: return jsonify({"success":False,"source":"CISA KEV","message":str(e),"vulnerabilities":[]}),502

@app.get("/api/threat-research/notifications")
@require_auth
def threat_notifications(user):
    con=db(); high=con.execute("SELECT id,event_type,timestamp,risk_score,detail FROM authentication_events WHERE user_id=? AND risk_score>=71 ORDER BY id DESC LIMIT 30",(user["id"],)).fetchall(); actions=con.execute("SELECT id,event_type,timestamp,description FROM audit_logs WHERE user_id=? AND event_type IN ('Detection Triaged','Detection Resolved','Case Created') ORDER BY id DESC LIMIT 30",(user["id"],)).fetchall(); con.close(); out=[{"id":r["id"],"event_type":r["event_type"],"timestamp":r["timestamp"],"risk_score":r["risk_score"],"detail":r["detail"],"type":"THREAT"} for r in high]+[{"id":r["id"],"event_type":r["event_type"],"timestamp":r["timestamp"],"risk_score":None,"detail":r["description"],"type":"CASE"} for r in actions]; out.sort(key=lambda x:x["timestamp"],reverse=True); return jsonify({"success":True,"notifications":out[:50]})

@app.get("/api/tenant")
@require_auth
def tenant(user):
    con=db(); rows=con.execute("SELECT id,name,tenant_key,status,notes,created_at FROM tenants WHERE user_id=? ORDER BY id DESC",(user["id"],)).fetchall(); ev=con.execute("SELECT COUNT(*) c FROM authentication_events WHERE user_id=?",(user["id"],)).fetchone()["c"]; cases=con.execute("SELECT COUNT(*) c FROM fraud_alerts WHERE user_id=?",(user["id"],)).fetchone()["c"]; con.close(); return jsonify({"success":True,"tenant":{"name":"AI-ISF Owner Tenant","owner":public_user(user),"users":1,"events":ev,"cases":cases,"data_mode":"REAL TELEMETRY ONLY"},"tenants":[dict(r) for r in rows]})

@app.post("/api/tenant")
@require_auth
def create_tenant(user):
    data=request.get_json(silent=True) or {}; name=(data.get("name") or "").strip()[:80]; notes=(data.get("notes") or "").strip()[:500]
    if len(name)<2:return jsonify({"success":False,"message":"Tenant name is required."}),400
    key="TEN-"+secrets.token_hex(4).upper(); con=db(); cur=con.execute("INSERT INTO tenants(user_id,name,tenant_key,status,notes,created_at) VALUES(?,?,?,?,?,?)",(user["id"],name,key,"ACTIVE",notes,now())); con.commit(); tid=cur.lastrowid; con.close(); audit(user["id"],"Tenant Created",f"Tenant {name} ({key}) created"); return jsonify({"success":True,"tenant_id":tid,"tenant_key":key})

@app.get("/api/tools/my-ip")
@require_auth
def my_ip(user):
    ip=request.headers.get("X-Forwarded-For",request.remote_addr or "Unknown").split(",")[0].strip(); return jsonify({"success":True,"ip":ip,"note":"This is the IP visible to the AI-ISF server. Behind NAT/proxy it may differ from your public internet IP."})

@app.get("/api/tools/virustotal")
@require_auth
def virustotal(user):
    indicator=(request.args.get("indicator") or "").strip(); key=os.environ.get("VT_API_KEY")
    if not indicator:return jsonify({"success":False,"message":"Enter an IP, domain, URL or file hash."}),400
    if not key:return jsonify({"success":False,"configured":False,"message":"VT_API_KEY is not configured. Add your VirusTotal API v3 key to the environment; the key is never stored in the browser."}),503
    import requests
    try:
        r=requests.get("https://www.virustotal.com/api/v3/search",params={"query":indicator},headers={"x-apikey":key},timeout=12); data=r.json(); audit(user["id"],"CTI Lookup",f"VirusTotal lookup performed for indicator type {indicator[:80]}"); return jsonify({"success":r.ok,"configured":True,"status_code":r.status_code,"data":data})
    except Exception as e:return jsonify({"success":False,"configured":True,"message":f"VirusTotal request failed: {e}"}),502

@app.get("/api/tools/rdap")
@require_auth
def rdap(user):
    domain=(request.args.get("domain") or "").strip().lower();
    if not domain:return jsonify({"success":False,"message":"Enter a domain."}),400
    import requests
    try:
        r=requests.get(f"https://rdap.org/domain/{domain}",timeout=12); audit(user["id"],"CTI Lookup",f"RDAP lookup for {domain}"); return jsonify({"success":r.ok,"status_code":r.status_code,"data":r.json()})
    except Exception as e:return jsonify({"success":False,"message":str(e)}),502

if __name__ == "__main__":
    init_db()
    print("\n"+"="*64); print(" AI-ISF — AI-Driven Identity Authentication & Fraud Prevention"); print(" http://127.0.0.1:5000"); print(" OpenCV:",cv2.__version__); print(" SQLite:",DB_PATH); print(" ML: Isolation Forest"); print(" Face recognition:","SFace + YuNet READY" if sface_recognizer is not None else "UNAVAILABLE (models will download on first start)"); print("="*64+"\n")
    app.run(debug=True,host="0.0.0.0",port=5000)
