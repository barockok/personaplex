"""Modal deployment for PersonaPlex multi-session server.

Setup:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx HUGGING_FACE_HUB_TOKEN=hf_xxx

Deploy:
    modal deploy modal_app.py
"""

import modal

app = modal.App("personaplex")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libopus-dev", "pkg-config", "build-essential")
    .pip_install(
        "numpy>=1.26,<2.2",
        "safetensors>=0.4.0,<0.5",
        "huggingface-hub>=0.24,<0.25",
        "einops==0.7",
        "sentencepiece==0.2",
        "sphn>=0.1.4,<0.2",
        "aiohttp>=3.10.5,<3.11",
        "prometheus-client>=0.20,<1.0",
    )
    .pip_install(
        "torch>=2.2.0,<2.5",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"PYTHONPATH": "/app/moshi"})
    .add_local_dir("moshi", "/app/moshi")
)


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
)
@modal.web_server(port=8998, startup_timeout=300)
def server():
    """Start PersonaPlex server in a subprocess."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/moshi"

    # Single subprocess — Modal's web_server decorator waits for port 8998
    # to become ready (up to startup_timeout=300s)
    subprocess.Popen(
        [
            sys.executable, "-m", "moshi.server",
            "--host", "0.0.0.0",
            "--port", "8998",
            "--max-sessions", "3",
            "--worker-id", "modal-a10g",
        ],
        env=env,
    )
