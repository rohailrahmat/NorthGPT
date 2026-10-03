import os, secrets, traceback
from functools import wraps
from flask import (
    Flask, jsonify, render_template, request,
    session, redirect, url_for
)
from werkzeug.exceptions import HTTPException
import rag, db

BASE = os.path.dirname(os.path.abspath(__file__))
SERVERLESS = bool(os.environ.get("VERCEL"))

STATIC = os.path.join(BASE, "public", "static")
if not os.path.isdir(STATIC):
    STATIC = os.path.join(BASE, "static")

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
# Set SECRET_KEY in your environment. The dev fallback is for local use only.
app.secret_key = os.environ.get("SECRET_KEY") or (
    secrets.token_hex(32) if SERVERLESS else "dev-only-secret-change-me")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SERVERLESS,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
)

# ── Helpers ───────────────────────────────────────────────────────────
def current_user():
    uid = session.get("user_id")
    return db.get_user_by_id(uid) if uid else None

def login_user(user):
    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]

def api_auth(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*a, **k)
    return wrapper

@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return e
    if app.debug:
        raise e
    traceback.print_exc()  # shows in Vercel function logs
    if request.path.startswith("/api/"):
        return jsonify({"error": "Server error: " + str(e)}), 500
    return "Something went wrong. Please try again.", 500

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ── Pages ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return render_template(
        "landing.html",
        user=current_user(),
        n=len(rag.DEST),
        destinations=list(rag.DEST.values()),
        guides=list(rag.GUIDES.values()),
        n_guides=len(rag.GUIDES)
    )

@app.get("/app")
@app.get("/chat")
def chat_app():
    user = current_user()
    if not user:
        session.clear()
        return redirect(url_for("login"))
    return render_template(
        "index.html",
        user=user,
        chat_sessions=db.get_user_chat_sessions(user["id"]),
        n=len(rag.DEST),
        guides=list(rag.GUIDES.values()),
        n_guides=len(rag.GUIDES)
    )

# ── Auth ──────────────────────────────────────────────────────────────
@app.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("chat_app"))
    return render_template("login.html")

@app.post("/login")
def login_post():
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    user, err = db.authenticate_user(email, password)
    if err:
        return render_template("login.html", error=err, email=email)
    login_user(user)
    return redirect(url_for("chat_app"))

@app.get("/register")
def register():
    if session.get("user_id"):
        return redirect(url_for("chat_app"))
    return render_template("register.html")

@app.post("/register")
def register_post():
    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    phone    = request.form.get("phone", "").strip()
    form = dict(name=name, email=email, phone=phone)
    if not name or not email or not password:
        return render_template("register.html", error="All fields are required.", **form)
    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters.", **form)
    user, err = db.register_user(name, email, password, phone)
    if err:
        return render_template("register.html", error=err, **form)
    login_user(user)
    return redirect(url_for("chat_app"))

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))

# ── Chat Sessions API ─────────────────────────────────────────────────
@app.post("/api/sessions/new")
@api_auth
def new_session():
    return jsonify(db.create_chat_session(session["user_id"]))

@app.get("/api/sessions")
@api_auth
def list_sessions():
    return jsonify(db.get_user_chat_sessions(session["user_id"]))

@app.get("/api/sessions/<sid>")
@api_auth
def get_session(sid):
    data = db.get_chat_session(sid, session["user_id"])
    if not data:
        return jsonify({"error": "Not found"}), 404
    return jsonify(data)

@app.delete("/api/sessions/<sid>")
@api_auth
def delete_session(sid):
    db.delete_chat_session(sid, session["user_id"])
    return jsonify({"ok": True})

@app.patch("/api/sessions/<sid>/title")
@api_auth
def rename_session(sid):
    title = ((request.get_json(silent=True) or {}).get("title") or "").strip()
    if title:
        db.update_session_title(sid, session["user_id"], title)
    return jsonify({"ok": True})

# ── Chat API ──────────────────────────────────────────────────────────
@app.post("/api/chat")
@api_auth
def chat():
    uid = session["user_id"]
    j   = request.get_json(silent=True) or {}
    q   = (j.get("q") or "").strip()
    sid = (j.get("session_id") or "").strip()
    compare = bool(j.get("compare", False))
    if not q:
        return jsonify({"error": "Empty query"}), 400

    # Load or create the session, then read its history once
    if sid:
        sess_data = db.get_chat_session(sid, uid)
        if not sess_data:
            return jsonify({"error": "Session not found"}), 404
        history = [{"role": m["role"], "content": m["content"]}
                   for m in (sess_data.get("messages") or [])]
    else:
        sid = db.create_chat_session(uid, q[:50])["id"]
        history = []

    if not history:
        db.update_session_title(sid, uid, q[:55])

    # Run RAG first, so a failed answer does not leave an unanswered message
    try:
        result = rag.answer(q, compare=compare, history=history)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    db.add_chat_message(sid, "user", q)
    db.add_chat_message(
        sid, "assistant", result["text"],
        sources=result.get("sources", []),
        pins=result.get("pins", []),
        guides=result.get("guides", [])
    )
    result["session_id"] = sid
    return jsonify(result)

# ── Other APIs ────────────────────────────────────────────────────────
@app.post("/api/itinerary")
@api_auth
def itin():
    j = request.get_json(silent=True) or {}
    try:
        days, budget = int(j["days"]), int(j["budget"])
        interests = (j.get("interests") or "sightseeing").strip()
        return jsonify(rag.itinerary(days, budget, interests))
    except (KeyError, ValueError):
        return jsonify({"error": "Send days, budget and interests."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.get("/api/dest/<i>")
def dest(i):
    return jsonify(rag.DEST.get(i, {}))

@app.get("/api/guides")
def guides_api():
    return jsonify(list(rag.GUIDES.values()))

@app.get("/api/guide/<gid>")
def guide(gid):
    return jsonify(rag.GUIDES.get(gid, {}))

@app.get("/api/me")
@api_auth
def api_me():
    return jsonify(current_user())

if __name__ == "__main__":
    app.run(debug=True)