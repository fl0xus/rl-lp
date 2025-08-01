# rl-lp

## Inference Server

We use vllm as an inference server. The provided docker-compose.yaml

### vllm docker compose

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.10.0
    container_name: vllm_oai
    ipc: host
    restart: "unless-stopped"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    ports:
      - 5801:8000
    command:
      ["--model", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "--max_model_len", "8000", "--gpu-memory-utilization", "0.5", "--dtype", "auto"]
```

## Benchmarks

### ORQA

The benchmarks for ORQA can be reproduced by first running the script `0_create_llm_answers.py`. If you want to output the exact matches from the benchmark run use `1_evaluate_answers.py`.

To do this, the model and inference server must be adjusted in the code. Change lines 5-9 (example below) according to your setup in `0_create_llm_answers.py`. Replace `$INFERENCE_ENDPOINT` with a your `url/v1` or `<ip>:<port>/v1`. Change `MODEL_NAME` to the correct model name. In this example, it is `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. If you don't know the correct name you can look it up at `$INFERENCE_ENDPOINT/v1/models`

```python
client = AsyncOpenAI(
    base_url="$INFERENCE_ENDPOINT/v1"
)

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
```

It is necessary to install the openai package with pip. You may need a virtual environment for this.

On Mac / Linux you can create and use a virtual environment with:

```bash
# create virtual environment
python -m venv venv

# use virtual environment
source venv/bin/activate

# install openai python package
pip install openai
```

After these steps, the benchmark can be run using

```bash
python 0_create_llm_answers.py

python 1_evaluate_answers.py orqa_run.jsonl
```





### ComplexOR

//TODO
