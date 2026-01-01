from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=False,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

system_message = (
    "Bạn là chuyên gia tư vấn tâm lý. "
    "Luôn trả lời bằng tiếng Việt, giọng nhẹ nhàng, đồng cảm, "
    "không phán xét, tập trung lắng nghe và hỗ trợ tinh thần."
    "Không sử dụng tiếng Anh trừ khi người dùng yêu cầu."
)

# ✅ CHỈ LẤY TEXT – LOẠI HTML / UI
def extract_text(content):
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if item.get("type") == "text"
        )
    return ""

def stream_response(message, history):
    history_langchain_format = [SystemMessage(content=system_message)]

    # ✅ GIỚI HẠN HISTORY (CHỐNG PHÌNH TOKEN)
    MAX_TURNS = 6
    history = history[-MAX_TURNS:]

    for msg in history:
        role = msg["role"]
        text = extract_text(msg["content"])

        if not text.strip():
            continue

        if role == "user":
            history_langchain_format.append(HumanMessage(content=text))
        elif role == "assistant":
            history_langchain_format.append(AIMessage(content=text))

    if message:
        history_langchain_format.append(HumanMessage(content=message))

        partial = ""
        for chunk in llm.stream(history_langchain_format):
            if chunk.content:
                partial += chunk.content
                yield partial


with gr.Blocks(
    css="""
    body {
        background: linear-gradient(135deg, #e6f4f1, #f7fbfa);
        font-family: 'Segoe UI', sans-serif;
    }

    .header {
        text-align: center;
        padding: 24px;
    }

    .header h1 {
        color: #1f4f4f;
        font-size: 36px;
        margin-bottom: 6px;
    }

    .header p {
        color: #4f6f6f;
        font-size: 16px;
    }

    .chatbot {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        padding: 10px;
    }

    textarea {
        border-radius: 14px !important;
        padding: 12px !important;
        font-size: 15px !important;
    }
    """
) as demo:

    gr.Markdown(
        """
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/387/387561.png" width="72"/>
            <h1>Tư vấn tâm lý</h1>
            <p>🌿 Lắng nghe – Thấu hiểu – Đồng hành cùng bạn</p>
        </div>
        """
    )

    gr.ChatInterface(
        fn=stream_response,
        chatbot=gr.Chatbot(
            height=480,
            elem_classes="chatbot"
        ),
        textbox=gr.Textbox(
            placeholder="Hãy chia sẻ điều bạn đang cảm thấy...",
            scale=7
        ),
    )

demo.launch(debug=True, share=False)
