import os
import json
import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="$INFERENCE_ENDPOINT/v1"
)

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

INPUT_FILES = [
    ("ORQA_test.jsonl", "ORQA_test"),
    ("ORQA_validation.jsonl", "ORQA_validation")
]
OUTPUT_FILE = "orqa_run.jsonl"
BATCH_SIZE = 10


def format_options(options):
    """
    Formats a list of options into a numbered string list.
    """
    formatted = []
    for idx, opt in enumerate(options, start=0):
        formatted.append(f"{idx}. {opt.strip()}")
    return "\n".join(formatted)


def read_inputs(files):
    """
    Reads JSONL files and constructs list of dicts with ORIGIN, PROMPT, TARGET_ANSWER
    """
    data = []
    for filename, origin in files:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                context = obj.get("CONTEXT", "").strip()

                # format and label the options
                options_list = obj.get("OPTIONS", [])
                if isinstance(options_list, list):
                    formatted_opts = format_options(options_list)
                else:
                    formatted_opts = options_list.strip()
                options = f"Options\n{formatted_opts}"

                # label the question
                question_text = obj.get("QUESTION", "").strip()
                question = f"Question\n{question_text}"

                # put question before options
                prompt = "\n\n".join([context, question, options])

                data.append({
                    "ORIGIN": origin,
                    "PROMPT": prompt,
                    "TARGET_ANSWER": obj.get("TARGET_ANSWER")
                })
    return data



async def infer_batch(batch):
    """
    Send a batch of prompts concurrently, returning their single-token completions
    """
    tasks = []

    for item in batch:
        task = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are only allowed to answer with a digit from the Options."},
                {"role": "user", "content": item["PROMPT"]}
                ],
            stream=False,
            temperature=0,
            max_tokens=1
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content.strip() for r in responses]


async def main():
    data = read_inputs(INPUT_FILES)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            llm_answers = await infer_batch(batch)
            for record, answer in zip(batch, llm_answers):
                record["LLM_ANSWER"] = answer
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Inference complete, results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
