from flask import (
    Flask, jsonify, render_template, request,
    session, redirect, url_for
)
import rag, db

app = Flask(__name__)
app.secret_key = "northgpt-secret-key-2025-change-in-prod"

# ── Auth helpers ──────────────────────────────────────────────────────
def current_user():
    uid = session.get("user_id")
    return db.get_user_by_id(uid) if uid else None

def require_login():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return None

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
    redir = require_login()
    if redir:
        return redir
    user = current_user()
    sessions = db.get_user_chat_sessions(user["id"])
    return render_template(
        "index.html",
        user=user,
        chat_sessions=sessions,
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
    session["user_id"]   = user["id"]
    session["user_name"] = user["name"]
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
    if not name or not email or not password:
        return render_template("register.html", error="All fields are required.", name=name, email=email, phone=phone)
    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters.", name=name, email=email, phone=phone)
    user, err = db.register_user(name, email, password, phone)
    if err:
        return render_template("register.html", error=err, name=name, email=email, phone=phone)
    session["user_id"]   = user["id"]
    session["user_name"] = user["name"]
    return redirect(url_for("chat_app"))

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ── Chat Sessions API ─────────────────────────────────────────────────
@app.post("/api/sessions/new")
def new_session():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    s = db.create_chat_session(session["user_id"])
    return jsonify(s)

@app.get("/api/sessions")
def list_sessions():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(db.get_user_chat_sessions(session["user_id"]))

@app.get("/api/sessions/<sid>")
def get_session(sid):
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    data = db.get_chat_session(sid, session["user_id"])
    if not data:
        return jsonify({"error": "Not found"}), 404
    return jsonify(data)

@app.delete("/api/sessions/<sid>")
def delete_session(sid):
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    db.delete_chat_session(sid, session["user_id"])
    return jsonify({"ok": True})

@app.patch("/api/sessions/<sid>/title")
def rename_session(sid):
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    title = (request.get_json() or {}).get("title", "").strip()
    if title:
        db.update_session_title(sid, session["user_id"], title)
    return jsonify({"ok": True})

# ── Chat API ──────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    j   = request.get_json() or {}
    q   = j.get("q", "").strip()
    sid = j.get("session_id", "").strip()
    compare = j.get("compare", False)
    if not q:
        return jsonify({"error": "Empty query"}), 400

    # Load or create session
    if not sid:
        s = db.create_chat_session(session["user_id"], q[:50])
        sid = s["id"]
    else:
        if not db.get_chat_session(sid, session["user_id"]):
            return jsonify({"error": "Session not found"}), 404

    # Load history from DB
    sess_data = db.get_chat_session(sid, session["user_id"])
    history = [{"role": m["role"], "content": m["content"]} for m in (sess_data.get("messages") or [])]

    # Auto-title if first message
    if len(history) == 0:
        db.update_session_title(sid, session["user_id"], q[:55])

    # Save user message
    db.add_chat_message(sid, "user", q)

    # Run RAG
    try:
        result = rag.answer(q, compare=compare, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save assistant message
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
def itin():
    j = request.get_json() or {}
    try:
        return jsonify(rag.itinerary(j["days"], j["budget"], j["interests"]))
    except Exception as e:
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
def me():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(current_user())

if __name__ == "__main__":
    app.run(debug=True)
