import gradio as gr
from transformers import pipeline

# Load model yang sudah di-train
coder = pipeline("text2text-generation", model="your-username/my-ai-coder")

def generate(instruction):
    result = coder(f"Write code: {instruction}", max_length=100)
    return result[0]['generated_text']

gr.Interface(
    fn=generate,
    inputs=gr.Textbox(lines=2, placeholder="Describe function..."),
    outputs=gr.Code(language="python"),
    title="My AI Coder"
).launch()
