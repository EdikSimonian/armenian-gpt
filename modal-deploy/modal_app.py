"""
ArmGPT Modal deployment.

Hosts the 320M Armenian GPT on a T4 GPU with scale-to-zero. Provides:
  - POST /generate_stream  — SSE streaming, simple JSON request
  - POST /v1/chat/completions  — OpenAI-compatible chat endpoint (streaming + non-streaming)
  - POST /v1/completions       — OpenAI-compatible text completions

Deploy:    modal deploy modal_app.py
Hot-test:  modal serve  modal_app.py
"""

import json
import time
import uuid

import modal


# ---------------------------------------------------------------------------
# Image: torch CPU base + cuda runtime, sentencepiece, fastapi
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "sentencepiece",
        "fastapi[standard]",
        "huggingface_hub",
    )
    .add_local_python_source("model", "bpe_tokenizer")
)

# Persistent Modal Volume holding the checkpoint + tokenizer.
# Populated once via upload_to_volume.py
volume = modal.Volume.from_name("armgpt-checkpoints", create_if_missing=True)
VOLUME_MOUNT = "/ckpt"

app = modal.App("armgpt")

# Auth secret — change via: modal secret create armgpt-auth ARMGPT_API_KEY=<new-key>
auth_secret = modal.Secret.from_name("armgpt-auth")


# ---------------------------------------------------------------------------
# Inference class — one instance per warm container
# ---------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[auth_secret],
    scaledown_window=60,     # idle 60s → scale to zero (was 120)
    timeout=300,             # max 5 min per request
    max_containers=1,        # single container — queues requests rather than spawning parallel GPUs
)
class ArmGPT:
    @modal.enter()
    def setup(self):
        """Runs once per cold start. Loads model + tokenizer into VRAM."""
        import torch
        from model import GPT
        from bpe_tokenizer import BPETokenizer

        print("[setup] loading tokenizer...", flush=True)
        self.tokenizer = BPETokenizer.load(f"{VOLUME_MOUNT}/tokenizer.json")
        print(f"[setup] vocab_size = {self.tokenizer.vocab_size}", flush=True)

        print("[setup] loading checkpoint...", flush=True)
        ckpt = torch.load(
            f"{VOLUME_MOUNT}/checkpoint.pt",
            map_location="cuda",
            weights_only=False,
        )
        cfg = ckpt["config"]
        self.checkpoint_step = ckpt.get("step", "unknown")
        print(f"[setup] checkpoint step={self.checkpoint_step}, cfg={cfg.get('n_layer')}L/{cfg.get('n_embd')}d", flush=True)

        self.model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            dropout=0.0,
        ).to("cuda")

        # Strip torch.compile prefix if present
        state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
            print("[setup] stripped _orig_mod. prefix", flush=True)

        self.model.load_state_dict(state)
        self.model.eval()
        # Warm up cuda kernels with one tiny forward pass
        with torch.no_grad():
            self.model(torch.zeros((1, 4), dtype=torch.long, device="cuda"))
        print("[setup] ready", flush=True)

    @modal.method()
    def generate_stream(
        self,
        prompt: str,
        length: int = 200,
        temperature: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
    ):
        """Generator yielding text deltas as tokens are produced."""
        import torch
        import torch.nn.functional as F

        ids = self.tokenizer.encode(prompt)
        if not ids:
            yield ""
            return

        idx = torch.tensor([ids], dtype=torch.long, device="cuda")
        full_ids = list(ids)
        last_text = self.tokenizer.decode(full_ids)
        # Don't yield the prompt itself — clients already have it.

        with torch.no_grad():
            for _ in range(int(length)):
                idx_cond = idx[:, -self.model.block_size:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :]

                if repetition_penalty != 1.0:
                    seen = torch.unique(idx_cond)
                    seen_logits = logits[:, seen]
                    seen_logits = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )
                    logits[:, seen] = seen_logits

                logits = logits / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                full_ids.append(int(idx_next.item()))

                # Decode the full sequence each step (BPE deltas across token
                # boundaries are unreliable; full decode + diff is correct).
                new_text = self.tokenizer.decode(full_ids)
                delta = new_text[len(last_text):]
                if delta:
                    yield delta
                last_text = new_text

    @modal.method()
    def info(self) -> dict:
        return {
            "model": "edisimon/armgpt",
            "checkpoint_step": self.checkpoint_step,
            "vocab_size": self.tokenizer.vocab_size,
        }


# ---------------------------------------------------------------------------
# Web app (ASGI) — SSE + OpenAI-compatible endpoints
# ---------------------------------------------------------------------------
@app.function(image=image, max_containers=2, secrets=[auth_secret])
@modal.asgi_app()
def web():
    import os
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(title="ArmGPT", version="1.0")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    API_KEY = os.environ.get("ARMGPT_API_KEY", "")

    def verify_auth(request: Request):
        """Check Bearer token or x-api-key header. Raises 401 if invalid."""
        if not API_KEY:
            return  # no key configured → open (shouldn't happen in prod)
        auth = request.headers.get("Authorization", "")
        xkey = request.headers.get("x-api-key", "")
        token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else xkey
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @web_app.get("/")
    def root(request: Request):
        verify_auth(request)
        return {
            "model": "edisimon/armgpt",
            "endpoints": {
                "POST /generate_stream": "Simple SSE: {prompt, length, temperature, top_k, repetition_penalty}",
                "POST /v1/chat/completions": "OpenAI-compatible chat (stream=true|false)",
                "POST /v1/completions": "OpenAI-compatible completions (stream=true|false)",
                "GET /v1/models": "OpenAI-compatible model list",
                "GET /info": "Loaded checkpoint info",
            },
        }

    @web_app.get("/info")
    def info(request: Request):
        verify_auth(request)
        return ArmGPT().info.remote()

    # ----- Simple SSE endpoint (used by the HF Space frontend) -----
    @web_app.post("/generate_stream")
    async def generate_stream(req: Request):
        verify_auth(req)
        body = await req.json()
        prompt = body.get("prompt", "")
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt required")
        length = int(body.get("length", 200))
        temperature = float(body.get("temperature", 0.85))
        top_k = int(body.get("top_k", 40))
        repetition_penalty = float(body.get("repetition_penalty", 1.15))

        def event_stream():
            try:
                for delta in ArmGPT().generate_stream.remote_gen(
                    prompt, length, temperature, top_k, repetition_penalty
                ):
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ----- OpenAI compat: /v1/models -----
    @web_app.get("/v1/models")
    def list_models(request: Request):
        verify_auth(request)
        return {
            "object": "list",
            "data": [{
                "id": "armgpt",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "edisimon",
            }],
        }

    def _messages_to_prompt(messages: list) -> str:
        """Flatten an OpenAI chat messages array into a single prompt string."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _params(body: dict) -> dict:
        return dict(
            length=int(body.get("max_tokens", 200)),
            temperature=float(body.get("temperature", 0.85)),
            top_k=int(body.get("top_k", 40)),
            repetition_penalty=float(body.get("repetition_penalty", 1.15)),
        )

    # ----- OpenAI compat: /v1/chat/completions -----
    @web_app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        verify_auth(req)
        body = await req.json()
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")
        prompt = _messages_to_prompt(messages)
        stream = bool(body.get("stream", False))
        params = _params(body)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            def sse():
                # Initial chunk with role
                first = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": "armgpt",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first)}\n\n"
                try:
                    for delta in ArmGPT().generate_stream.remote_gen(prompt, **params):
                        chunk = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": "armgpt",
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as e:
                    err = {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(err)}\n\n"
                # Final stop chunk
                final = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": "armgpt",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming: aggregate and return
        full = ""
        for delta in ArmGPT().generate_stream.remote_gen(prompt, **params):
            full += delta
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "armgpt",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })

    # ----- OpenAI compat: /v1/completions -----
    @web_app.post("/v1/completions")
    async def completions(req: Request):
        verify_auth(req)
        body = await req.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt required")
        stream = bool(body.get("stream", False))
        params = _params(body)
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            def sse():
                try:
                    for delta in ArmGPT().generate_stream.remote_gen(prompt, **params):
                        chunk = {
                            "id": completion_id, "object": "text_completion",
                            "created": created, "model": "armgpt",
                            "choices": [{"text": delta, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as e:
                    err = {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(err)}\n\n"
                final = {
                    "id": completion_id, "object": "text_completion",
                    "created": created, "model": "armgpt",
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        full = ""
        for delta in ArmGPT().generate_stream.remote_gen(prompt, **params):
            full += delta
        return JSONResponse({
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": "armgpt",
            "choices": [{
                "text": full, "index": 0, "logprobs": None, "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })

    return web_app
