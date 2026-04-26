FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# --- Layer 1: PyTorch CPU-only (~200 MB vs ~2 GB for CUDA) ----------------
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# --- Layer 2: oncallenv runtime dependencies ------------------------------
RUN pip install --no-cache-dir \
    openenv-core>=0.2.0 \
    pydantic>=2.5.0 \
    pyyaml>=6.0.0

# --- Layer 3: Gradio app + ML inference deps ------------------------------
COPY files/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Layer 4: Pre-download base model (cached ~6 GB) ----------------------
ENV HF_HOME=/app/hf_cache
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16)"

# --- Layer 5: Model adapter + scenario data (rarely changes) --------------
COPY models/feedback-model-cp300/ models/feedback-model-cp300/
COPY scenarios_seed/ scenarios_seed/
COPY curriculum_results/ curriculum_results/

# --- Layer 6: Source code + Gradio app (rebuilt on code changes, fast) -----
COPY src/ src/
COPY files/ files/

ENV PYTHONPATH="/app/src:$PYTHONPATH"

EXPOSE 7860

CMD ["python", "files/app.py"]
