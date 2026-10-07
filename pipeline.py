"""Meeting assistant pipeline: Whisper -> LLM #1 (refiner) -> LLM #2 (secretary)."""
import difflib
import json
import os
import re
import time

from openai import OpenAI, RateLimitError

# ---------- Config (override with environment variables) ----------
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")  # hosted Whisper (Groq)
WHISPER_SIZE = STT_MODEL  # alias used by app.py caption
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
REFINER_MODEL = os.getenv("REFINER_MODEL", "llama3.1:8b")  # LLM #1
WRITER_MODEL = os.getenv("WRITER_MODEL", "qwen2.5:7b")     # LLM #2
MAX_MB = int(os.getenv("MAX_MB", "25"))
MIN_WORDS = 30
REFINE_PASSES = int(os.getenv("REFINE_PASSES", "2"))
WRITER_PART_WORDS = int(os.getenv("WRITER_PART_WORDS", "1800"))
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".mp4"}


class UserError(Exception):
    """An error whose message is safe and friendly to show the user."""


_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


# ---------- Checks (cheap first) ----------
UNREADABLE = "We couldn't read this file. It may be damaged. Please try another file."


def _looks_like_audio(head):
    """Check the file's real content (first bytes), not just its name."""
    return (
        head[:3] == b"ID3"                                                   # mp3 with tag
        or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)  # raw mp3/aac frame
        or (head[:4] == b"RIFF" and head[8:12] == b"WAVE")                   # wav
        or head[4:8] == b"ftyp"                                              # m4a / mp4
        or head[:4] == b"fLaC"                                               # flac
        or head[:4] == b"OggS"                                               # ogg
        or head[:4] == b"\x1a\x45\xdf\xa3"                                   # webm
    )


def validate_file(path, filename, size_bytes):
    if not filename:
        raise UserError("Please upload an audio file first.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise UserError("This file type isn't supported. Please upload an audio file such as MP3, WAV or M4A.")
    if size_bytes == 0:
        raise UserError("This file is empty. Please upload a recording that has audio in it.")
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except Exception:
        raise UserError(UNREADABLE)
    if not _looks_like_audio(head):
        raise UserError(UNREADABLE)
    if size_bytes > MAX_MB * 1024 * 1024:
        raise UserError(f"This recording is larger than we can process (limit: {MAX_MB} MB). "
                        "Please upload a smaller or more compressed file, such as MP3.")


# ---------- Stage 1: transcription (hosted Whisper) ----------
def _g(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _speech_text(resp):
    segs = _g(resp, "segments") or []
    kept = []
    for s in segs:
        # Drop segments Whisper itself thinks are not speech (silence hallucinations).
        if (_g(s, "no_speech_prob", 0) or 0) > 0.6 and (_g(s, "avg_logprob", 0) or 0) < -1.0:
            continue
        t = (_g(s, "text", "") or "").strip()
        if t:
            kept.append(t)
    if kept:
        return " ".join(kept)
    return "" if segs else (_g(resp, "text", "") or "").strip()


def transcribe(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        resp = client().audio.transcriptions.create(
            file=(os.path.basename(path), data),
            model=STT_MODEL,
            language="en",
            temperature=0,
            response_format="verbose_json",
            prompt="A business meeting with technical terms, acronyms, names and numbers.",
        )
    except Exception as e:
        msg = str(e).lower()
        if isinstance(e, RateLimitError) or "429" in msg or "rate limit" in msg:
            raise UserError(f"The speech service reached its limit. Please wait {_wait_text(_retry_seconds(e))} "
                            "for it to refresh, then try again.")
        if "401" in msg or "api key" in msg:
            raise UserError("The speech service rejected the API key. Please check LLM_API_KEY.")
        if "400" in msg or "could not process" in msg or "invalid" in msg:
            raise UserError(UNREADABLE)
        raise UserError("Transcription failed. Please check your internet connection and try again.")
    text = _speech_text(resp)
    if not text:
        raise UserError("No speech was detected in this recording.")
    if len(text.split()) < MIN_WORDS:
        raise UserError("The recording is too short to create meeting minutes.")
    return text


# ---------- LLM helper ----------
def _retry_seconds(err):
    """Seconds the service says to wait ('try again in 7.66s', '1m23s', '2h3m'), or None."""
    m = re.search(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s)\s*)+)", str(err))
    if not m:
        return None
    units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(n) * units[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)))


def _wait_text(sec):
    if sec is None or sec < 60:
        return "about a minute"
    if sec < 3600:
        return f"about {int(sec // 60) + 1} minutes"
    h = int(sec // 3600) + 1
    return f"about {h} hour" + ("s" if h > 1 else "")


def chat(model, system, user, temperature=0.1, effort=None, fallback=None):
    kwargs = {}
    if effort and "gpt-oss" in model:
        kwargs["extra_body"] = {"reasoning_effort": effort}
    r = None
    for attempt in range(5):
        try:
            r = client().chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
            break
        except RateLimitError as e:
            wait = _retry_seconds(e)
            # Short wait: wait for the per-minute token window to refresh, then retry.
            # Long wait (e.g. a daily limit): do not block; use the backup model if there is one.
            if (wait is not None and wait > 75) or attempt == 4:
                if fallback and fallback != model:
                    print(f"[fallback] {model} is rate limited ({_wait_text(wait)}); using {fallback}")
                    return chat(fallback, system, user, temperature, effort)
                raise
            time.sleep((wait if wait is not None else 15 * (attempt + 1)) + 1)
    content = r.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("empty model reply")
    return content.strip()


# ---------- Stage 2: refinement (LLM #1) ----------
REFINE_PROMPT = """You are a careful transcript editor for technology and business meetings.
The text is an automatic speech-to-text transcript, so some words were misheard.
Your job is to FIX misrecognized words, using the surrounding context to decide what was really said.
Fix these kinds of errors:
- Technical terms, tools, products and acronyms heard as similar-sounding ordinary words or nonsense
  (for example "cooper nettys" -> "Kubernetes", "post gres" -> "PostgreSQL", "O auth" -> "OAuth", "A W S" -> "AWS").
- A real word that makes no sense in context but sounds like the technical word that does
  (for example "catching" in a talk about a Redis cache means "caching"; "mitigation" in a talk about moving a database means "migration").
- Obvious homophone slips where only one reading fits the context (for example "to" vs "two").
- Read every sentence for grammar and logic. A word that makes a sentence ungrammatical or illogical is probably misheard:
  replace it with the sound-alike that fits (for example "the server slows down about 1,000 users" means "above 1,000 users").
- Casing and light punctuation.
If a word is a close sound-alike of the term that makes sense in context, FIX it. Do not be timid about clear cases.
NEVER change:
- the meaning, negations ("not", "don't", "no"), commitments, or who does what;
- numbers and dates (keep the digits exactly as they are);
- people's names. A capitalized word followed by a comma (like "Name, we will...") is usually a person being addressed:
  keep it exactly as written, even if it looks odd. Never replace a name with another word such as "OK".
Spoken grammar is not an error. Never add or remove small words (the, a, to), never change pronouns (I, me, myself, we, you),
never smooth, rephrase or complete sentences. Only replace a misheard word with a similar-sounding word that fits.
Do not summarize, shorten, reorder, or add content. Keep the speaker's wording otherwise, including filler phrases.
Output ONLY the corrected transcript text, with no comments."""


def _chunks(text, max_words=1000):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


_COMMON = set("""so okay ok yeah yes no well now then and but or the a an we i it he she they you this that these those
there here what when where why how who let lets if as at in on for with next first second also alright right sure thanks thank
hello hi please maybe just actually anyway great good cool all some someone everyone nobody one two our my your their his her
is are was were do does did can could should would will have has had not dont didnt im ill were thats its""".split())


def _is_name_like(w):
    """A capitalized word that is not an ordinary sentence word: a person, a day, a month, a product name."""
    c = re.sub(r"[^A-Za-z']", "", w)
    return len(c) > 2 and c[0].isupper() and c.lower().replace("'", "") not in _COMMON


_KEY = set("""i me my mine myself we us our ours ourselves you your yours he him his she her hers they them their theirs
not no never none cannot dont cant wont isnt arent wasnt werent didnt doesnt shouldnt wouldnt couldnt havent hasnt
will ill shall should must maybe might probably can could would im ive agree agreed decide decided""".split())


def _tok(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _keys(words):
    """Words that carry meaning and must survive editing: pronouns, negation, commitments, numbers."""
    out = []
    for w in words:
        t = _tok(w)
        if t in _KEY or any(c.isdigit() for c in t):
            out.append(t)
    return sorted(out)


def _protect_names(original, fixed):
    """Keep only safe edits: undo changes to names, pronouns, negation, commitments or numbers, and drop pure insertions."""
    a, b = original.split(), fixed.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = a[i1:i2], b[j1:j2]
        if tag == "equal":
            out += new
        elif tag == "insert":
            continue  # the refiner may fix words, not add new ones
        elif _norm(" ".join(old)) == _norm(" ".join(new)):
            out += new  # only casing/punctuation changed
        elif any(_is_name_like(w) for w in old) and not any(_is_name_like(w) for w in new):
            out += old
        elif _keys(old) != _keys(new):
            out += old
        else:
            out += new
    return " ".join(out)


SECOND_LOOK = """
This text has already been through one editing pass. Read it again sentence by sentence as a grammar and logic check.
Fix any word that still makes a sentence illogical or technically wrong because it was misheard, by swapping in a similar-sounding word. Do not rephrase anything.
If nothing needs fixing, return the text unchanged."""


def refine(raw):
    """Returns (refined_text, warning_or_None). Whatever succeeded is kept; a failure never loses the transcript."""
    rejected, problem, stop = False, None, False
    parts = []
    for ch in _chunks(raw):
        cur = ch
        for i in range(0 if stop else max(1, REFINE_PASSES)):
            prompt = REFINE_PROMPT if i == 0 else REFINE_PROMPT + SECOND_LOOK
            try:
                try:
                    proposed = chat(REFINER_MODEL, prompt, cur, effort="medium", fallback=WRITER_MODEL)
                except RateLimitError:
                    raise
                except Exception as e1:  # transient error or empty reply: one quick retry
                    print(f"[refine] retrying after {type(e1).__name__}: {e1}")
                    proposed = chat(REFINER_MODEL, prompt, cur, effort="medium", fallback=WRITER_MODEL)
            except RateLimitError as e:
                print(f"[refine error] rate limit: {e}")
                problem, stop = problem or ("limit", e), True
                break
            except Exception as e:
                print(f"[refine error] {type(e).__name__}: {e}")  # visible in the terminal for debugging
                problem = problem or ("error", e)
                break
            out = _protect_names(ch, proposed)
            print(f"[refine] pass {i + 1}: model proposed {len(list_changes(cur, proposed))} edits, "
                  f"{len(list_changes(cur, out))} kept after the safety guard")
            n_in, n_out = len(cur.split()), len(out.split())
            # Length guard: the editor must neither drop nor add content.
            if n_out < 0.85 * n_in or n_out > 1.15 * n_in:
                rejected = True
                continue
            cur = out
        parts.append(cur)
    text = " ".join(parts)
    if problem and problem[0] == "limit":
        warn = ("The language-model service reached its token limit, so part of the transcript is shown as transcribed (not refined). "
                f"Wait {_wait_text(_retry_seconds(problem[1]))} for the limit to refresh, then press Retry below.")
    elif problem:
        warn = "Refinement could not be completed for part of the transcript, so that part is shown as transcribed. Press Retry below to try again."
    elif rejected:
        warn = "Part of the refinement was rejected by a safety check (the length changed too much), so the earlier text was kept for that part."
    else:
        warn = None
    return text, warn


def list_changes(raw, refined):
    """Word-level differences between raw and refined, as (old, new) pairs."""
    a, b = raw.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if _norm(old) != _norm(new):  # ignore pure case/punctuation changes
            out.append((old, new))
    return out


# ---------- Stage 3: minutes, decisions, tasks (LLM #2) ----------
RECORD_PROMPT = """You are a meeting secretary. Read the transcript and return ONLY valid JSON (no markdown) with this shape:
{
  "summary": "3-5 sentence summary",
  "minutes": [{"topic": "...", "points": ["...", "..."]}],
  "decisions": [{"decision": "...", "evidence": "exact short quote from the transcript"}],
  "action_items": [{"task": "clear sentence of the work to be done", "owner": null, "deadline": null, "evidence": "exact short quote"}]
}
STRICT RULES:
- A decision is something the participants clearly AGREED on. Suggestions like "maybe we could" or "we should think about" are NOT decisions; put them in minutes as "discussed".
- List EVERY task that someone says must be done or commits to doing, even when nobody owns it yet. An unassigned task such as "someone has to update the documentation" MUST appear in action_items with owner null. Do not turn mere ideas or proposals into tasks.
- owner: fill ONLY when the transcript clearly says who will do the task. Addressing a person ("Meera, ...") does not make them the owner unless the sentence assigns the task to them. If it is ambiguous, or the speaker takes the task themselves without giving their name, use null.
- If a sentence contradicts itself about who does the task (for example a named person plus "myself"), the owner is unclear, so use null.
- deadline: fill ONLY if a date or time is explicitly stated for that task, otherwise null.
- Never invent names, numbers, dates, or topics. Use only what is in the transcript.
- Keep numbers and limits exactly as meant (for example "handled 40,000 but not stable above 50,000" must not become "stable up to 40,000").
- Keep the summary and minutes consistent with action_items: never name a person as responsible for a task in the summary or minutes unless that person is the owner of that task in action_items. If the owner is null, write "someone" or leave the person out.
- Keep minutes concise and organised by topic.
- If there are no decisions or no tasks, return an empty list."""


def _parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())


def _verify(record, transcript):
    """Code-level guards against invention."""
    t = _norm(transcript)
    for d in record.get("decisions", []):
        d["evidence_found"] = bool(d.get("evidence")) and _norm(d["evidence"]) in t
    for a in record.get("action_items", []):
        owner = a.get("owner")
        if not owner or _norm(owner) not in t:
            a["owner"] = None
        dl = a.get("deadline")
        if dl:
            tokens = [w for w in _norm(dl).split() if len(w) > 2]
            if not tokens or not any(w in t for w in tokens):
                a["deadline"] = None
        a["evidence_found"] = bool(a.get("evidence")) and _norm(a["evidence"]) in t
    return record


def _record_for(text, full_text):
    for _ in range(2):  # one retry if JSON is broken
        try:
            rec = _parse_json(chat(WRITER_MODEL, RECORD_PROMPT, text, fallback=REFINER_MODEL))
            for k, default in (("summary", ""), ("minutes", []), ("decisions", []), ("action_items", [])):
                rec.setdefault(k, default)
            return _verify(rec, full_text)
        except RateLimitError as e:
            print(f"[record error] rate limit: {e}")
            raise UserError("The language-model service reached its token limit. "
                            f"Please wait {_wait_text(_retry_seconds(e))} for it to refresh, then try again.")
        except Exception as e:
            print(f"[record error] {e!r}")
    raise UserError("We couldn't generate the meeting record. Please try again.")


MERGE_SUMMARY_PROMPT = ("Combine these partial summaries of consecutive parts of ONE meeting into a single summary of 3-5 sentences. "
                        "Use only the information given. Keep numbers exactly. Do not say anyone is responsible for a task unless it is stated. "
                        "Output only the summary.")


def generate_record(refined):
    if len(refined.split()) <= WRITER_PART_WORDS * 1.15:
        return _record_for(refined, refined)
    # Long meeting: write the record part by part, then merge (keeps each request small).
    recs = [_record_for(part, refined) for part in _chunks(refined, WRITER_PART_WORDS)]
    topics, seen = {}, set()
    merged = {"summary": "", "minutes": [], "decisions": [], "action_items": []}
    for r in recs:
        for m in r["minutes"]:
            key = _norm(m.get("topic"))
            if key in topics:
                topics[key]["points"] = topics[key].get("points", []) + m.get("points", [])
            else:
                topics[key] = m
        for name, field in (("decisions", "decision"), ("action_items", "task")):
            for item in r[name]:
                k = (name, _norm(item.get(field)))
                if k not in seen:
                    seen.add(k)
                    merged[name].append(item)
    merged["minutes"] = list(topics.values())
    try:
        merged["summary"] = chat(WRITER_MODEL, MERGE_SUMMARY_PROMPT, "\n\n".join(r["summary"] for r in recs))
    except Exception:
        merged["summary"] = " ".join(r["summary"] for r in recs)
    return merged


# ---------- Human-readable output ----------
def record_to_markdown(rec):
    L = ["# Meeting Record", "", "## Summary", rec["summary"], "", "## Minutes"]
    for m in rec["minutes"]:
        L.append(f"### {m.get('topic', '')}")
        L += [f"- {p}" for p in m.get("points", [])]
    L += ["", "## Key Decisions"]
    L += [f"- {d['decision']}" for d in rec["decisions"]] or ["- None recorded"]
    L += ["", "## Action Items"]
    for a in rec["action_items"]:
        L.append(f"- {a['task']} | Owner: {a.get('owner') or 'Unspecified'} | Deadline: {a.get('deadline') or 'Unspecified'}")
    if not rec["action_items"]:
        L.append("- None recorded")
    return "\n".join(L)
