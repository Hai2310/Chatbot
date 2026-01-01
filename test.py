from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022",streaming=True)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


system_message = (
    "Bạn là chuyên gia tư vấn tâm lý. "
    "Luôn trả lời bằng tiếng Việt, giọng thân thiện, dễ hiểu. "
    "Không sử dụng tiếng Anh trừ khi người dùng yêu cầu."
)

def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    history_langchain_format = [
        SystemMessage(content=system_message)
    ]

    # ✅ Gradio v4 history
    for msg in history:
        role = msg["role"]
        text = msg["content"][0]["text"]

        if role == "user":
            history_langchain_format.append(HumanMessage(content=text))
        elif role == "assistant":
            history_langchain_format.append(AIMessage(content=text))

    if message:
        history_langchain_format.append(HumanMessage(content=message))

        partial_message = ""
        for chunk in llm.stream(history_langchain_format):
            if chunk.content:
                partial_message += chunk.content
                yield partial_message



demo_interface = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
                       container=False,
                       autoscroll=True,
                       scale=7),
)

demo_interface.launch(debug=True , share = True)