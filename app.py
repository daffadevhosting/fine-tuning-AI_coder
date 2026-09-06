"""
Daffa AI Coder — Multi-language Code Generator
Model: https://huggingface.co/sendaljepit/daffa-ai-coder-multilang
"""

import gradio as gr
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

# Load once at startup (CPU Space is fine for CodeT5-small)
coder = pipeline(
    "text2text-generation",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    max_new_tokens=256,
    num_beams=3,
)


def generate_code(instruction: str, language: str) -> str:
    if not instruction or not str(instruction).strip():
        return "# Please enter a description first."
    prompt = f"Generate {language} code: {str(instruction).strip()}"
    out = coder(prompt)
    return out[0]["generated_text"]


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
    outputs=gr.Code(language="python", label="Generated Code"),
    title="\ud83d\udc0d Daffa AI Coder",
    description=(
        "Multi-language code generation from natural language. "
        "Model: [sendaljepit/daffa-ai-coder-multilang](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang) "
        "(fine-tuned CodeT5-small). "
        "Languages: Python, JavaScript, TypeScript, Java, SQL, Go, PHP, HTML, CSS, C++, Rust."
    ),
    examples=[
        ["Write a function to add two numbers", "python"],
        ["Write a function to reverse a string", "javascript"],
        ["Write a SQL query to count orders per customer", "sql"],
        ["Write a simple struct for a Person", "go"],
        ["Write a method to calculate factorial", "java"],
        ["Write CSS to center a div", "css"],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
