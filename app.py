"""
Daffa AI Coder — ZeroGPU-compatible Gradio Space
Model: https://huggingface.co/sendaljepit/daffa-ai-coder-multilang
"""

import gradio as gr
import spaces
from transformers import pipeline

MODEL_PATH = "sendaljepit/daffa-ai-coder-multilang"

SUPPORTED_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "sql",
    "go",
    "php",
    "html",
    "css",
    "cpp",
    "rust",
]

_pipe = None


def get_pipe():
    """Load model only when GPU is available (inside @spaces.GPU)."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text2text-generation",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            max_new_tokens=256,
            num_beams=3,
            device_map="auto",
        )
    return _pipe


@spaces.GPU(duration=60)
def generate_code(instruction: str, language: str) -> str:
    if not instruction or not str(instruction).strip():
        return "# Please enter a description first."

    prompt = f"Generate {language} code: {str(instruction).strip()}"
    result = get_pipe()(prompt)
    return result[0]["generated_text"]


demo = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(
            lines=4,
            label="Describe what you want",
            placeholder="Example: Write a function to reverse a string",
        ),
        gr.Dropdown(
            choices=SUPPORTED_LANGUAGES,
            value="python",
            label="Programming Language",
        ),
    ],
    outputs=gr.Textbox(lines=16, label="Generated Code"),
    title="\ud83d\udc0d Daffa AI Coder",
    description=(
        "Multi-language code generation (fine-tuned CodeT5-small). "
        "Model: [sendaljepit/daffa-ai-coder-multilang]"
        "(https://huggingface.co/sendaljepit/daffa-ai-coder-multilang). "
        "Runs on **ZeroGPU** \u2014 first request may be slower while the model loads."
    ),
    examples=[
        ["Write a function to add two numbers", "python"],
        ["Write a function to reverse a string", "javascript"],
        ["Write a SQL query to count orders per customer", "sql"],
        ["Write a simple struct for a Person", "go"],
        ["Write a method to calculate factorial", "java"],
        ["Write CSS to center a div", "css"],
    ],
    cache_examples=False,
)

# HF Spaces + ZeroGPU: do not set share=True
demo.queue(max_size=10)
demo.launch()
