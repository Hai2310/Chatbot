import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# =========================
# Load local model
# =========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# =========================
# System prompt
# =========================
SYSTEM_PROMPT = """
Bạn là một chuyên gia tư vấn tâm lý.
Giọng nói nhẹ nhàng, tôn trọng, không phán xét.
Không đưa ra chẩn đoán y khoa.
Luôn khuyến khích người dùng chia sẻ cảm xúc.
"""

# =========================
# Chat function
# =========================
def stream_response(message, history):
    prompt = SYSTEM_PROMPT + "\n"

    for user, bot in history:
        prompt += f"Người dùng: {user}\nTrợ lý: {bot}\n"

    prompt += f"Người dùng: {message}\nTrợ lý:"

    output = generator(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    text = output[0]["generated_text"]
    answer = text[len(prompt):].strip()

    # streaming giả
    partial = ""
    for ch in answer:
        partial += ch
        yield partial


# =========================
# Gradio UI
# =========================
demo = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Bạn đang cảm thấy thế nào?",
        container=False,
        scale=7
    ),
    title="🧠 Chatbot Tư Vấn Tâm Lý (Local)",
    description="Chạy hoàn toàn trên máy – không cần Internet",
)

demo.launch()
