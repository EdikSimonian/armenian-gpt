# Modal deployment for ArmGPT

GPU-backed inference endpoint with auth. T4, scale-to-zero.

## Live endpoint
`https://edisimon--armgpt-web.modal.run`

## Auth
All endpoints require `Authorization: Bearer <key>` or `x-api-key: <key>`.
The key is stored as a Modal secret named `armgpt-auth` — never in code.

## Setup on a new machine
```bash
pip install modal
modal token new   # browser auth
cp model.py modal-deploy/
cp tokenizers/bpe_tokenizer.py modal-deploy/
cd modal-deploy
python upload_to_volume.py        # push latest HF checkpoint to Modal volume
modal deploy modal_app.py         # deploy
```

## Rotate API key (no redeploy needed)
```bash
modal secret create armgpt-auth ARMGPT_API_KEY=<new-key>
modal app stop armgpt && modal deploy modal_app.py
```

## Update checkpoint
```bash
python upload_to_volume.py [step_NNNNN]
modal app stop armgpt && modal deploy modal_app.py
```
