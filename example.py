import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("D:\models\Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path)
    # max_tokens太差会报错
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "你是谁",
        "中国是",
        "讲一个笑话，题材任意，200字",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"len {len(output['text'])} Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
