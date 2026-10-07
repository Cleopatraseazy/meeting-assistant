# Technical Description: AI Meeting Assistant

## Pipeline
Audio upload -> **Stage 1: speech-to-text** -> raw transcript -> **Stage 2: refinement LLM** -> refined transcript -> **Stage 3: documentation LLM** -> structured meeting record (Markdown + JSON).

The stages always run in this order, as one workflow, from a single upload.

## Models and roles
| Stage | Model | Role |
|---|---|---|
| 1. Transcription | Whisper large-v3-turbo (`whisper-large-v3-turbo`, served by Groq) | Converts the English meeting audio into the raw transcript. Forced to English, deterministic decoding (temperature 0), segments that Whisper itself marks as non-speech are dropped. |
| 2. Refinement (LLM #1) | `openai/gpt-oss-20b` | Acts as a transcript editor. Fixes plausible recognition errors in technical terms, acronyms and obvious homophones, using context. Never changes names, numbers, negation or commitments. |
| 3. Documentation (LLM #2) | `openai/gpt-oss-120b` | Acts as a meeting secretary. Reads the refined transcript only and produces the summary, organised minutes, key decisions and action items as JSON. |

The two language-model roles are separate stages with separate prompts and separate model settings (`REFINER_MODEL`, `WRITER_MODEL`).

## How outputs move between stages
1. Stage 1 returns plain text. If it is empty or fewer than 30 words, the pipeline stops with a clear message, so no LLM ever sees empty text.
2. The raw transcript is split into chunks of about 1000 words and sent to Stage 2. Each chunk is edited in two passes (the second is a grammar-and-logic re-read). The raw transcript is kept unchanged for display and download.
3. The refined transcript (not the raw one) is sent to Stage 3, which returns JSON containing summary, minutes, decisions (each with an evidence quote) and action items (task, owner, deadline).
   Long meetings are written part by part and merged, so each request stays within the free-plan token limit; rate-limit errors are retried automatically.
4. The app renders the JSON on screen and offers a Markdown version and a JSON version. Both carry the same decisions and tasks.

## Safeguards against invention
- Prompts define a decision as something clearly agreed. Proposals ("maybe we could...") go into the minutes as discussed, not into decisions.
- Owner and deadline are `null` unless explicitly stated, and are shown as "Unspecified".
- Code checks after generation: an owner whose name does not appear in the transcript is reset to Unspecified; a deadline whose words do not appear is reset; every decision's evidence quote is checked against the transcript and flagged if not found.
- Refinement guards: output length must stay within 15% of the input, otherwise the original text is kept; edits that rewrite an addressed name are reverted by code. The interface lists every change the refiner made, so each fix can be inspected.
- If refinement fails, the raw transcript is used and the user is told. If record generation fails, both transcripts are still shown.

## Error handling
Handled with a friendly message: no file, unsupported type, empty file, unreadable or fake audio (file content is checked, not only the name), file too large, no speech detected, recording too short, and failure of any stage. The status area shows which step is running or which step failed.

Rate limits: on a short limit the app waits for the time the service gives and retries; on a long limit (for example a daily cap) the refinement and record stages switch to the other configured model; if no model is available the user sees how long to wait.

## Run it
See `README.md`.
