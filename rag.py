import json, re, time, logging, numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("northgpt")
client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
# Ordered fallback chain:
GEN_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemma-4-31b-it"
]
FLOOR = 0.50

DEST = {d["id"]: d for d in json.load(open("data/destinations.json", encoding="utf-8"))}
GUIDES = {g["id"]: g for g in json.load(open("data/guides.json", encoding="utf-8"))}

QUIET = re.compile(r"quiet|offbeat|less crowded|hidden|peaceful|remote", re.I)
GUIDE_QUERY = re.compile(r"guide|tour operator|driver|jeep hire|book|contact|whatsapp|phone|hire|agency|porter|call", re.I)
GREETING_QUERY = re.compile(r"^(hi|hello|hey|salam|assalam|aoa|who are you|what can you do|help me|how are you|good morning|good evening|good afternoon)[\s!.,?]*$", re.I)

SYSTEM = """You are NorthGPT, an advanced AI travel intelligence assistant specialized in Gilgit-Baltistan, Pakistan (the Karakoram, Himalayas, and Hindu Kush).
Your responses reflect the caliber of modern flagship LLMs (like Claude and ChatGPT): direct, articulate, professional, and precise.

Key Operating Principles:
1. Direct & Specific:
   - Answer the user's specific question directly and accurately without generic boilerplate, repetitive introductory fluff, or unsolicited data dumps.
   - Focus precisely on what the traveler asked.
   - If the user sends a simple greeting (e.g. "Hi", "Hello", "Salam"), respond in 1-2 brief, professional sentences asking how you can help them with Gilgit-Baltistan travel.
2. Grounded Factuality:
   - Base all geographical facts, altitudes, seasons, road conditions, and PKR costs strictly on verified knowledge and the provided Context.
   - State exact numbers clearly (e.g. altitude in meters, best travel months, estimated costs in PKR).
   - If an inquiry is out of region, impossible, or absurd (e.g. asking for a jeep to the Moon), politely and succinctly state the scope limitation without unnecessary lecture.
3. Targeted Guide & Logistics Linking:
   - Only recommend specific certified local guides/tour operators if the user asks for guides, booking, transport, or if the trek strictly requires a certified guide.
4. Clean, Professional Markdown:
   - Use clear formatting with bold key figures, structured bullet points for multi-part information, and concise paragraphs."""

# ── Retry + Fallback ──────────────────────────────────────────────
RETRYABLE = {429, 500, 502, 503, 504}

def _generate_with_retry(contents, config=None, max_attempts=2):
    """Try each model in GEN_MODELS with exponential backoff on retryable errors, and fallback on 404/503."""
    last_err = None
    for model in GEN_MODELS:
        for attempt in range(max_attempts):
            try:
                kwargs = dict(model=model, contents=contents)
                if config:
                    kwargs["config"] = config
                result = client.models.generate_content(**kwargs)
                if model != GEN_MODELS[0]:
                    log.info("Used fallback model: %s", model)
                return result
            except Exception as e:
                code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if code == 404 or "not found" in str(e).lower() or "no longer available" in str(e).lower():
                    log.warning("Model %s not available (%s), trying next fallback model...", model, e)
                    last_err = e
                    break
                elif code in RETRYABLE:
                    wait = 1.0 * (attempt + 1)
                    log.warning("Model %s returned %s, retrying in %ss (attempt %d/%d)",
                                model, code, wait, attempt + 1, max_attempts)
                    time.sleep(wait)
                    last_err = e
                else:
                    raise  # non-retryable error — bubble up immediately
        log.warning("Model %s exhausted, switching to next model...", model)
    raise last_err or RuntimeError("All models and retries exhausted with no recorded error.")

def make_chunks():
    out = []
    # 1. Destination Chunks
    for d in DEST.values():
        parts = {
            "overview": f"{d['name']} ({d['district']}). {d['overview'] or ''}",
            "how_to_reach": d.get("how_to_reach"),
            "season_safety": " ".join(
                x for x in [
                    f"Best season: {d['best_season']}." if d.get("best_season") else "",
                    f"Altitude: {d['altitude_m']} m." if d.get("altitude_m") else "",
                    d.get("safety") or ""
                ] if x
            ),
            "activities": ", ".join(d["activities"]) if d.get("activities") else None,
            "tips": d.get("tips"),
            "costs": f"Estimated cost per day: {d['cost_per_day_pkr']} PKR." if d.get("cost_per_day_pkr") else None
        }
        for sec, t in parts.items():
            if t:
                out.append({
                    "dest": d["id"],
                    "type": "dest",
                    "section": sec,
                    "text": f"{d['name']} - {sec}: {t}",
                    "hidden": d.get("hidden_gem", False)
                })

    # 2. Tour Guide Chunks
    for g in GUIDES.values():
        guide_text = (
            f"Certified Tour Guide & Operator: {g['name']} ({g['company']}) for {g['region']} ({', '.join(g['districts'])}). "
            f"Specialties: {', '.join(g['specialties'])}. License: {g['license']}. WhatsApp: {g['whatsapp']}. Phone: {g['phone']}. "
            f"Languages: {', '.join(g['languages'])}. Daily fee: ~{g['daily_rate_pkr']} PKR/day. Bio: {g['bio']}"
        )
        out.append({
            "dest": g["id"],
            "type": "guide",
            "guide_id": g["id"],
            "section": "tour_guide",
            "text": guide_text,
            "hidden": False
        })

    return out

def embed(texts):
    r = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    v = np.array([e.values for e in r.embeddings])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def build_index():
    ch = make_chunks()
    log.info("Building vector index for %d chunks...", len(ch))
    # Batch embeddings in chunks of 50 to avoid payload limits
    vecs_list = []
    batch_size = 50
    for i in range(0, len(ch), batch_size):
        batch = [c["text"] for c in ch[i:i + batch_size]]
        vecs_list.append(embed(batch))
    vecs = np.vstack(vecs_list)
    for c, v in zip(ch, vecs):
        c["vec"] = v.tolist()
    json.dump(ch, open("data/index.json", "w", encoding="utf-8"))
    log.info("Successfully indexed %d chunks into data/index.json", len(ch))

_idx = None
def retrieve(q, k=5):
    global _idx
    if _idx is None:
        ch = json.load(open("data/index.json", encoding="utf-8"))
        _idx = (ch, np.array([c["vec"] for c in ch]))
    ch, V = _idx
    q_vec = embed([q])[0]
    s = V @ q_vec
    
    # Boost hidden gems if requested
    if QUIET.search(q):
        s = s + np.array([0.05 if c.get("hidden") else 0 for c in ch])
    
    # Boost guide chunks if user query mentions guide/driver/booking/contact
    if GUIDE_QUERY.search(q):
        s = s + np.array([0.08 if c.get("type") == "guide" else 0 for c in ch])

    top = np.argsort(-s)[:k]
    return [ch[i] for i in top], float(s[top[0]])

def refs(chunks, q=""):
    seen_dest = set()
    src = []
    pins = []
    guide_ids_matched = set()

    is_guide_query = bool(GUIDE_QUERY.search(q))

    for c in chunks:
        dest_id = c.get("dest")
        c_type = c.get("type", "dest")

        if c_type == "guide" or dest_id in GUIDES:
            g_id = c.get("guide_id") or dest_id
            guide_ids_matched.add(g_id)
        elif dest_id in DEST:
            d = DEST[dest_id]
            src.append({"dest": d["id"], "section": c.get("section", "overview"), "label": d["name"]})
            if d["id"] not in seen_dest:
                seen_dest.add(d["id"])
                pins.append({"id": d["id"], "name": d["name"], "lat": d["lat"], "lng": d["lng"]})
            
            # If user explicitly asked about guides or trek logistics, connect relevant district guides
            if is_guide_query:
                d_dist = d.get("district", "").lower()
                for g in GUIDES.values():
                    if any(dist.lower() in d_dist or d_dist in dist.lower() for dist in g.get("districts", [])):
                        guide_ids_matched.add(g["id"])

    # If query is explicitly asking for guides or booking, include top guides even if not directly referenced
    if is_guide_query and not guide_ids_matched:
        for g in list(GUIDES.values())[:3]:
            guide_ids_matched.add(g["id"])

    # Only include guides if this was a guide/booking query or guide chunks were directly matched
    matched_guides = [GUIDES[gid] for gid in guide_ids_matched if gid in GUIDES] if is_guide_query or any(c.get("type") == "guide" for c in chunks) else []
    return src, pins, matched_guides

def answer(q, compare=False, history=None):
    """Conversational, direct, state-of-the-art LLM answer with multi-turn memory."""
    # Build conversation context from history
    convo_history_str = ""
    if history and isinstance(history, list):
        turns = []
        for h in history[-6:]:  # Keep last 6 conversational turns
            role = "Traveler" if h.get("role") == "user" else "Assistant"
            content = h.get("content", "").strip()
            if content:
                turns.append(f"{role}: {content}")
        if turns:
            convo_history_str = "Conversation History:\n" + "\n".join(turns) + "\n\n"

    # Check for direct conversational greetings
    is_greeting = bool(GREETING_QUERY.match(q.strip()))
    
    plain = None
    if compare:
        try:
            plain = _generate_with_retry(q).text
        except Exception:
            plain = None

    # Handle simple greetings concisely without loading unrelated pins or guides
    if is_greeting:
        prompt = (
            f"{convo_history_str}"
            f"User: {q}\n\n"
            "Respond with a brief, professional, welcoming message (1-2 sentences) as NorthGPT, "
            "and ask how you can assist with their Gilgit-Baltistan travel plans."
        )
        r = _generate_with_retry(
            prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM)
        )
        return {
            "text": r.text,
            "sources": [],
            "pins": [],
            "guides": [],
            "plain": plain
        }

    # Retrieve relevant chunks from vector index
    try:
        chunks, best = retrieve(q)
    except Exception as e:
        log.warning("Vector retrieval error: %s", e)
        chunks, best = [], 0.0

    src, pins, matched_guides = refs(chunks, q)

    # Context assembly
    ctx_parts = []
    for c in chunks:
        label = c.get("dest")
        sec = c.get("section", "data")
        ctx_parts.append(f"[{label}/{sec}] {c.get('text', '')}")
    
    if matched_guides:
        for g in matched_guides[:3]:
            ctx_parts.append(
                f"[guide/{g['id']}] Guide: {g['name']} ({g['company']}) · WhatsApp: {g['whatsapp']} · Phone: {g['phone']} · Region: {g['region']} · Specialties: {', '.join(g['specialties'])}"
            )

    ctx = "\n\n".join(ctx_parts)

    # If low relevance and not a guide query, handle cleanly
    if best < FLOOR and not ctx and not GUIDE_QUERY.search(q):
        prompt = (
            f"{convo_history_str}"
            f"User Query: {q}\n\n"
            "State clearly and politely that you specialize in Gilgit-Baltistan travel intelligence, "
            "and invite them to ask about destinations (Hunza, Skardu, Fairy Meadows, etc.), routes, altitudes, seasons, or logistics in the region."
        )
        r = _generate_with_retry(
            prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM)
        )
        return {
            "text": r.text,
            "sources": [],
            "pins": [],
            "guides": [],
            "plain": plain
        }

    # Clean, direct, professional RAG generation
    prompt = (
        f"{convo_history_str}"
        f"Verified Knowledge Context:\n{ctx}\n\n"
        f"Traveler Query: {q}\n\n"
        "Instructions:\n"
        "- Give a direct, professional, and accurate response that answers exactly what the traveler asked.\n"
        "- Do not include generic fluff, repetitive excitement, or unrelated facts.\n"
        "- Use exact facts from the Context (altitude in meters, best season months, PKR costs, road conditions, safety precautions).\n"
        "- If the traveler specifically asked for guides or booking contacts, provide the certified guide details."
    )

    r = _generate_with_retry(
        prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM)
    )

    return {
        "text": r.text,
        "sources": src,
        "pins": pins,
        "guides": matched_guides,
        "plain": plain
    }

def itinerary(days, budget, interests):
    q = f"{interests} trip in Gilgit-Baltistan"
    chunks, _ = retrieve(q, k=8)
    
    # Also find guides matching the region/interests
    _, _, guides = refs(chunks, q)
    
    ctx = "\n\n".join(f"[{c['dest']}/{c.get('section', 'data')}] {c.get('text', '')}" for c in chunks)
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Plan a {days}-day trip in Gilgit-Baltistan with a total budget of {budget} PKR. Interests: {interests}. "
        "Use ONLY destinations in the context. Return JSON adhering to this exact format:\n"
        '{"days":[{"day":1,"dest_id":"...","title":"...","plan":"...","cost_pkr":number_or_null}],'
        '"safety_note":"...","recommended_guide":"..."}.\n'
        "cost_pkr must come from the context cost data, otherwise null. Never invent costs."
    )
    r = _generate_with_retry(
        prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json")
    )
    plan = json.loads(r.text)
    pins = []
    for d in plan.get("days", []):
        x = DEST.get(d.get("dest_id"))
        if x:
            pins.append({"id": x["id"], "name": x["name"], "lat": x["lat"], "lng": x["lng"]})
    plan["pins"] = pins
    plan["guides"] = guides[:3]
    return plan
