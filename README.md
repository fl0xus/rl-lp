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

1. Clone the repository Chain-of-Experts

```console
git clone https://github.com/xzymustbexzy/Chain-of-Experts.git
```

2. Create a virtual environment inside the directory

```console
cd Chain-of-Experts

python -m venv venv

source venv/bin/activate
```

3. The provided code only works for OpenAI models in the standard implementation. But you can modify the code in `Chain-of-Experts/experts/base_expert.py`. You have to exchange (lines 12-15)

```python
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=0
        )
```

to

```python
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=0,
            openai_api_base="$INFERENCE_ENDPOINT/v1",
            openai_api_key="test",
        )
```

Change `$INFERENCE_ENDPOINT` the same way as in ORQA benchmark above

Alternatively, simply take the file from `rl-lp/benchmarks/complex_or/base_expert.py` and replace `$INFERENCE_ENDPOINT`.

4. Install the required packages

```bash
pip install -r requirements.txt

pip install pulp
```

5. You can now run the benchmark with after replacing the model:

```bash
python run_exp.py \
  --dataset ComplexOR \
  --problem ".*" \
  --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
  --algorithm standard
```
