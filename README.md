# rl-lp

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