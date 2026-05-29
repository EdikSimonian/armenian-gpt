# ArmGPT giant-preset training image.
#
# Base: NGC PyTorch 25.01 — torch 2.6, CUDA 12.6, sm_120 (Blackwell) support,
# FlashAttention 2, Transformer Engine, Apex preinstalled.
#
# Build:  docker build -t armgpt:giant -f docker/Dockerfile .
# Run:    ./docker/run.sh   (mounts repo + HF cache, drops into bash)
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Project-specific deps not in the NGC base.
# hf_transfer extra enables parallelized downloads (needed for the 31 GB corpus).
RUN pip install --no-cache-dir \
    sentencepiece==0.2.0 \
    "huggingface_hub[hf_transfer]==0.27.1" \
    datasets==3.2.0 \
    zstandard==0.23.0 \
    mwxml==0.3.6

WORKDIR /workspace/armgpt

# Code and data are mounted at runtime via -v in run.sh, so the image stays
# small and code edits don't require a rebuild. No ENTRYPOINT — user controls
# when anything starts.
CMD ["bash"]
