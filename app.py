import html
import json
import os
import tempfile

import streamlit as st

import pipeline as p

st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@500;600&family=Public+Sans:wght@400;500;600&display=swap');
:root { --ink:#17202B; --muted:#5E6B7A; --line:#DDE2E8; --accent:#0E6E73; --soft:#E3F1F1; --warn:#9A5B00; --warnbg:#FFF4E0; --fail:#B3261E; }
.stApp { background:#F5F6F8; color:var(--ink); font-family:'Public Sans', system-ui, sans-serif; }
.block-container { max-width:1080px; padding-top:2.2rem; }
h1.app-title { font-family:'Newsreader', Georgia, serif; font-weight:600; font-size:2.6rem; letter-spacing:-0.01em; margin:0 0 .2rem; color:var(--ink); }
p.lede { color:var(--muted); font-size:1.05rem; max-width:620px; margin:0 0 1.4rem; }
h3.sec { font-family:'Newsreader', Georgia, serif; font-weight:600; font-size:1.35rem; margin:.4rem 0 .6rem; color:var(--ink); }
.steps { display:flex; gap:10px; margin:.2rem 0 1.2rem; flex-wrap:wrap; }
.step { flex:1; min-width:180px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 14px; color:var(--muted); font-weight:500; display:flex; align-items:center; gap:10px; }
.step .num { width:24px; height:24px; border-radius:50%; background:#EEF1F4; display:inline-flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:600; }
.step.active { border-color:var(--accent); color:var(--ink); } .step.active .num { background:var(--accent); color:#fff; }
.step.done { color:var(--ink); } .step.done .num { background:var(--soft); color:var(--accent); }
.step.failed { border-color:var(--fail); color:var(--fail); } .step.failed .num { background:var(--fail); color:#fff; }
.summary { background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:10px; padding:16px 20px; line-height:1.6; max-width:78ch; }
.decision { background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:10px; padding:12px 16px; margin-bottom:10px; }
.decision .d-text { font-weight:600; }
.decision .d-ev { color:var(--muted); font-size:.92rem; margin-top:4px; }
.decision .d-warn { color:var(--warn); background:var(--warnbg); display:inline-block; font-size:.85rem; padding:2px 8px; border-radius:6px; margin-top:6px; }
table.tasks { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
table.tasks th { text-align:left; background:#EEF1F4; padding:10px 14px; font-weight:600; font-size:.95rem; }
table.tasks td { padding:10px 14px; border-top:1px solid var(--line); vertical-align:top; }
td.unspec { color:var(--muted); font-style:italic; }
.empty { background:#fff; border:1px dashed var(--line); border-radius:10px; padding:14px 18px; color:var(--muted); }
[data-testid="stBaseButton-primary"], button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
[data-baseweb="tab-highlight"] { background-color:var(--accent); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STEP_NAMES = ["Transcribe audio", "Refine transcript", "Write meeting record"]


def steps_html(active=-1, failed=None, done=False):
    out = []
    for i, name in enumerate(STEP_NAMES):
        if failed == i:
            cls = "failed"
        elif done or i < active:
            cls = "done"
        elif i == active:
            cls = "active"
        else:
            cls = ""
        out.append(f'<div class="step {cls}"><span class="num">{i + 1}</span>{name}</div>')
    return '<div class="steps">' + "".join(out) + "</div>"


def esc(x):
    return html.escape(str(x or ""))


# ---------- Sidebar: models ----------
with st.sidebar:
    st.markdown("### Models used")
    st.markdown(f"**Transcription**  \n{p.STT_MODEL}")
    st.markdown(f"**Refinement (LLM #1)**  \n{p.REFINER_MODEL}")
    st.markdown(f"**Meeting record (LLM #2)**  \n{p.WRITER_MODEL}")
    st.caption("Each stage works only from the stage before it. Owners and deadlines that were not said stay Unspecified.")

# ---------- Header ----------
st.markdown('<h1 class="app-title">Meeting Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="lede">Turn a recorded meeting into a transcript, minutes, decisions and tasks, '
            'using only what was actually said.</p>', unsafe_allow_html=True)

up = st.file_uploader(f"Upload a meeting recording (English, MP3, WAV or M4A, up to {p.MAX_MB} MB)")
start = st.button("Start processing", type="primary")
st.caption("Free API plans limit how many tokens can be used per minute. If a limit message appears, wait for the time it shows, then run again.")
steps_ph = st.empty()

res = st.session_state.get("result")
if res and not start:
    steps_ph.markdown(steps_html(done=bool(res["record"]), failed=res.get("failed"), active=-1),
                      unsafe_allow_html=True)
else:
    steps_ph.markdown(steps_html(), unsafe_allow_html=True)

# ---------- Processing ----------
if start:
    st.session_state.pop("result", None)
    res = {"raw": None, "refined": None, "record": None, "warning": None, "error": None, "failed": None}
    status = st.status("Checking file...", expanded=False)
    stage_i = -1
    try:
        if up is None:
            raise p.UserError("Please upload an audio file first.")
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(up.getvalue())
            path = f.name
        p.validate_file(path, up.name, len(up.getvalue()))

        stage_i = 0
        steps_ph.markdown(steps_html(0), unsafe_allow_html=True)
        status.update(label="Transcribing audio...")
        res["raw"] = p.transcribe(path)

        stage_i = 1
        steps_ph.markdown(steps_html(1), unsafe_allow_html=True)
        status.update(label="Refining transcript...")
        res["refined"], res["warning"] = p.refine(res["raw"])

        stage_i = 2
        steps_ph.markdown(steps_html(2), unsafe_allow_html=True)
        status.update(label="Writing meeting record...")
        res["record"] = p.generate_record(res["refined"])

        steps_ph.markdown(steps_html(done=True), unsafe_allow_html=True)
        status.update(label="Done", state="complete", expanded=False)
    except p.UserError as e:
        step_label = STEP_NAMES[stage_i] if stage_i >= 0 else "Checking file"
        res["error"] = f"Failed at step '{step_label}': {e}"
        res["failed"] = stage_i if stage_i >= 0 else None
        status.update(label=res["error"], state="error")
    except Exception:
        step_label = STEP_NAMES[stage_i] if stage_i >= 0 else "Checking file"
        res["error"] = f"Failed at step '{step_label}': something unexpected went wrong. Please try again."
        res["failed"] = stage_i if stage_i >= 0 else None
        status.update(label=res["error"], state="error")
    if res["failed"] is not None:
        steps_ph.markdown(steps_html(failed=res["failed"], active=res["failed"]), unsafe_allow_html=True)
    st.session_state["result"] = res

# ---------- Results ----------
res = st.session_state.get("result")
if not res:
    st.markdown('<div class="empty">No recording processed yet. Upload a file above and choose '
                '<b>Start processing</b>. The transcript, minutes, decisions and tasks will appear here.</div>',
                unsafe_allow_html=True)
else:
    if res["error"]:
        st.error(res["error"])
    if res["warning"]:
        st.warning(res["warning"])

    if res["raw"] and (res["warning"] or not res["record"]):
        if st.button("Retry refinement and record"):
            with st.status("Retrying refinement and record...", expanded=False) as retry_status:
                try:
                    res["refined"], res["warning"] = p.refine(res["raw"])
                    res["record"] = p.generate_record(res["refined"])
                    res["error"], res["failed"] = None, None
                    retry_status.update(label="Done", state="complete")
                except p.UserError as e:
                    res["error"], res["failed"] = f"Failed at step 'Write meeting record': {e}", 2
                    retry_status.update(label=res["error"], state="error")
            st.session_state["result"] = res
            st.rerun()

    if res["raw"]:
        rec = res["record"]
        changes = p.list_changes(res["raw"], res["refined"]) if res["refined"] else []

        c1, c2, c3 = st.columns(3)
        c1.metric("Decisions", len(rec["decisions"]) if rec else "-")
        c2.metric("Action items", len(rec["action_items"]) if rec else "-")
        c3.metric("Fixes by the refiner", len(changes))

        if rec:
            st.markdown('<h3 class="sec">Summary</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary">{esc(rec["summary"])}</div>', unsafe_allow_html=True)

        st.markdown('<h3 class="sec">Downloads</h3>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Raw transcript (.txt)", res["raw"], "raw_transcript.txt", use_container_width=True)
        if res["refined"]:
            d2.download_button("Refined transcript (.txt)", res["refined"], "refined_transcript.txt", use_container_width=True)
        if rec:
            d3.download_button("Meeting record (.md)", p.record_to_markdown(rec), "meeting_record.md", use_container_width=True)
            d4.download_button("Meeting record (.json)", json.dumps(rec, indent=2), "meeting_record.json", use_container_width=True)

        st.write("")
        tabs = st.tabs(["Raw transcript", "Refined transcript", "Minutes", "Decisions", "Action items"])

        with tabs[0]:
            st.text_area("Raw transcript", res["raw"], height=320, label_visibility="collapsed")

        with tabs[1]:
            left, right = st.columns(2)
            left.markdown("**Raw**")
            left.text_area("raw copy", res["raw"], height=300, label_visibility="collapsed")
            right.markdown("**Refined**")
            right.text_area("refined", res["refined"] or "", height=300, label_visibility="collapsed")
            if changes:
                with st.expander(f"Changes made by the refiner ({len(changes)})", expanded=True):
                    for old_w, new_w in changes:
                        st.markdown(f"- ~~{old_w or '(nothing)'}~~ → **{new_w or '(removed)'}**")
            else:
                st.caption("The refiner made no changes to this transcript.")

        if rec:
            with tabs[2]:
                for m in rec["minutes"]:
                    st.markdown(f"**{m.get('topic', '')}**")
                    for pt in m.get("points", []):
                        st.markdown(f"- {pt}")
            with tabs[3]:
                if not rec["decisions"]:
                    st.markdown('<div class="empty">No decisions were recorded in this meeting.</div>', unsafe_allow_html=True)
                for d in rec["decisions"]:
                    warn = "" if d.get("evidence_found") else '<div class="d-warn">Quote not found word for word in the transcript</div>'
                    st.markdown(
                        f'<div class="decision"><div class="d-text">{esc(d.get("decision"))}</div>'
                        f'<div class="d-ev">From the recording: &ldquo;{esc(d.get("evidence"))}&rdquo;</div>{warn}</div>',
                        unsafe_allow_html=True)
            with tabs[4]:
                if not rec["action_items"]:
                    st.markdown('<div class="empty">No action items were recorded in this meeting.</div>', unsafe_allow_html=True)
                else:
                    rows = ""
                    for a in rec["action_items"]:
                        o, dl = a.get("owner"), a.get("deadline")
                        rows += (f'<tr><td>{esc(a.get("task"))}</td>'
                                 f'<td class="{"" if o else "unspec"}">{esc(o) if o else "Unspecified"}</td>'
                                 f'<td class="{"" if dl else "unspec"}">{esc(dl) if dl else "Unspecified"}</td></tr>')
                    st.markdown('<table class="tasks"><tr><th>Task</th><th>Owner</th><th>Deadline</th></tr>'
                                + rows + "</table>", unsafe_allow_html=True)
        else:
            for t in tabs[2:]:
                with t:
                    st.markdown('<div class="empty">The meeting record could not be generated. '
                                'The transcripts above are still available.</div>', unsafe_allow_html=True)
