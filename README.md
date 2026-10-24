# TidyTrace

A synthetic archive rehearsal built for DEV HF26 Week 3: Slow and Steady.
LangGraph 1.2.16 owns the state graph, checkpoint, interruption and resume.
The application previews one exact plan, takes a demonstration decision, applies
it to in-memory objects and can restore the exact original inventory.

No user files are read or changed. State and checkpoints disappear on exit.
Terminal decisions are not authenticated consent or contest submission receipts.

## Run

Use Python 3.12. From this folder on Windows:

```powershell
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.lock
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe -B -m unittest -v
& ./.venv/Scripts/python.exe -B demo.py
& ./.venv/Scripts/python.exe -B terminal_demo.py
```

The scripted demo checks real LangGraph interruption, a simulated decision and
exact undo. The 41 offline tests use injected requests and ephemeral loopback
HTTP fixtures; they do not call a model.

The terminal requires the exact plan ID for apply and receipt ID for undo.
Wrong input, EOF and piped input cannot approve. The included
[35-second video](docs/TIDYTRACE_DEMO.mp4) is a labelled replay of real captured
synthetic workflow states, with an automated demonstration decision.

## Optional model experiment

Scripted planning is the default. With an existing local Ollama installation and
the qwen2.5:14b model already available, `python model_demo.py --ollama` makes one
request using only the fixed synthetic fixture. It does not download a model,
retry or change endpoints. Existing receipt paths are refused before a request.

This export uses the loopback default `http://127.0.0.1:11434/api/generate`.
Its model mode has not been evaluated live at that destination. The earlier
private evaluations used another local runtime address, omitted from this export.
See [evaluation notes](EVALUATION.md) for the actual limited evidence.

The client subprocess has a total request budget. Ordinary timeout cleanup is
confirmed; exceptional cleanup failures are reported as uncertain. OS scheduling
is not a real-time guarantee. Closing the client does not establish cancellation
of server-side inference. Executor policy remains independent of model output.

## Scope and provenance

New application work began October 20, 2026. This is a synthetic prototype with
no production file adapter, persistence, authenticated consent, measured user
time savings or general model-reliability claim. No partner category is claimed.
No later competition-window commits are represented by this local export.

LangGraph's retained MIT notice is in third_party/LANGGRAPH_LICENSE.txt.
Pinned dependencies keep their own licences and are installed separately; their
source/binaries and model weights are not bundled. No application licence has
been selected or granted by preparing this local review candidate. The contest
rules do not mandate a particular application-source licence. Sharing rights
and final destination require the entrant's confirmation before publication.
