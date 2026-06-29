# Codex Restored Context

Restored on 2026-05-10 after Codex reconfiguration.

Project role: Vietnamese voice cloning tool based on VieNeu-TTS.

Related Codex session:
- 019e0201-7698-7401-a773-1b7ec3438069: "X?y tool clone voice Vi?t".

What exists:
- CLI runner: `run_voice_clone.bat`.
- Web UI runner: `run_voice_clone_ui.bat`, usually `http://127.0.0.1:7861`.
- API runner: `run_voice_clone_api.bat`, usually `http://127.0.0.1:8002`.
- Docs: `VOICE_CLONE_TOOL.md`, `VOICE_CLONE_API.md`.
- User voice metadata: `user_voices.json`.
- Output folders under `outputs/`.

Runtime state noticed during recovery:
- API process was listening on `127.0.0.1:8002`.

Current git status at recovery time:
- Branch `main`.
- Dirty working tree with modified VieNeu files and many untracked voice clone tool files.

Continue from here by reading `VOICE_CLONE_TOOL.md` and `VOICE_CLONE_API.md` first.
