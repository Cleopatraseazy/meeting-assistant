# AI Meeting Assistant

Turns a meeting recording into a raw transcript, a refined transcript, minutes, decisions and action items.

## Models and roles

|Stage|Model|Role|
|-|-|-|
|1|Whisper large-v3-turbo (hosted on Groq, `STT\_MODEL`)|Speech-to-text, raw transcript|
|2|LLM #1 (`REFINER\_MODEL`)|Fixes domain-specific recognition errors, preserves meaning|
|3|LLM #2 (`WRITER\_MODEL`)|Summary, minutes, decisions, action items (JSON)|

Data flow: audio -> raw transcript -> refined transcript -> structured record (Markdown + JSON).

## Setup

1. Python 3.10+
2. `pip install -r requirements.txt`
3. Create a free API key at console.groq.com. One key serves all three models.
4. Set environment variables (PowerShell shown; use `export` on Mac/Linux):

```
$env:LLM\_BASE\_URL="https://api.groq.com/openai/v1"
$env:LLM\_API\_KEY="your\_key"
$env:REFINER\_MODEL="openai/gpt-oss-20b"
$env:WRITER\_MODEL="openai/gpt-oss-120b"
```

Optional: `STT\_MODEL` (default `whisper-large-v3-turbo`), `MAX\_MB` (default 25).

## Run

```
python -m streamlit run app.py
```

Open http://localhost:8501, upload audio, click **Start processing**, then inspect and download the outputs.
Tip: keep recordings compressed (MP3/M4A), since the file-size limit is 25 MB.

## Error handling

No file, unsupported type, empty file, unreadable file, too long, no speech, too short, and failed stages all show a clear message. If refinement fails, the raw transcript is used. If record generation fails, the transcripts are still shown.

## Anti-invention safeguards

Prompts forbid guessing; owners/deadlines not found in the transcript are reset to "Unspecified"; each decision carries an evidence quote that is checked against the transcript.



\## Known limitations

\- Speech recognition can mishear names and drop punctuation; names are kept unchanged and owners stay "Unspecified" when ownership is unclear.

\- Summary and minutes are model-written and may occasionally phrase a number or ownership slightly differently from the action-items list.

\- Uploads are limited to 25 MB; use compressed formats like MP3 or M4A.

\- If a yellow note says the language-model service reached its token limit, wait the time it shows and press "Retry refinement and record" (no need to re-upload). Avoid running several recordings back to back.





