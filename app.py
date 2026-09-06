"""
Daffa AI Coder — Multi-language Code Generator
Powered by fine-tuned CodeT5
Model: https://huggingface.co/sendaljepit/daffa-ai-coder-multilang

Compatible with Hugging Face Spaces CPU and ZeroGPU.
"""

import gradio as gr
from transformers import pipeline

MODEL_PATH = "sendaljepit/daffa-ai-coder-multilang"

# Lazy-loaded pipeline (important for ZeroGPU)
_coder = None


def get_coder():
    global _coder
    if _coder is None:
        _coder = pipeline(
            "text2text-generation",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            max_new_tokens=256,
            num_beams=3,
        )
    return _coder


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

LANG_TO_HIGHLIGHT = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "sql": "sql",
    "go": "go",
    "php": "php",
    "html": "html",
    "css": "css",
    "cpp": "cpp",
    "rust": "rust",
}


def _generate(instruction: str, language: str) -> str:
    if not instruction or not instruction.strip():
        return "# Please enter a description first."

    prompt = f"Generate {language} code: {instruction.strip()}"
    result = get_coder()(prompt)
    return result[0]["generated_text"]


# ZeroGPU: decorate inference so GPU is allocated on demand
try:
    import spaces

    @spaces.GPU(duration=60)
    def generate_code(instruction: str, language: str) -> str:
        return _generate(instruction, language)

except ImportError:

    def generate_code(instruction: str, language: str) -> str:
        return _generate(instruction, language)


def update_highlight(language: str):
    return gr.Code(language=LANG_TO_HIGHLIGHT.get(language, "python"))


with gr.Blocks(title="Daffa AI Coder", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐍 Daffa AI Coder
        **Multi-language** code generation from natural language.

        Model: [`sendaljepit/daffa-ai-coder-multilang`](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang)
        (fine-tuned CodeT5-small)

        Supported: Python · JavaScript · TypeScript · Java · SQL · Go · PHP · HTML · CSS · C++ · Rust
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            language = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="python",
                label="Programming Language",
            )
            instruction = gr.Textbox(
                lines=4,
                placeholder="Example: Write a function to reverse a string",
                label="Describe what you want",
            )
            btn = gr.Button("Generate Code", variant="primary")

        with gr.Column(scale=1):
            output = gr.Code(
                language="python",
                label="Generated Code",
                lines=16,
            )

    language.change(fn=update_highlight, inputs=language, outputs=output)
    btn.click(fn=generate_code, inputs=[instruction, language], outputs=output)

    gr.Examples(
        examples=[
            ["Write a function to add two numbers", "python"],
            ["Write a function to reverse a string", "javascript"],
            ["Write a SQL query to count orders per customer", "sql"],
            ["Write a simple struct for a Person", "go"],
            ["Write a method to calculate factorial", "java"],
            ["Write CSS to center a div", "css"],
        ],
        inputs=[instruction, language],
    )

if __name__ == "__main__":
    demo.launch()
