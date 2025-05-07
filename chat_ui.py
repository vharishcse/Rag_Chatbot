# chat_ui.py
import os
import gradio as gr
from dotenv import load_dotenv
from rag_chatbot import generate_response

load_dotenv()

# Ensure Google API key is set
assert os.getenv("GOOGLE_API_KEY") is not None, "❌ GOOGLE_API_KEY not found in .env"

# Chat function with history
def chatbot(user_input, history):
    if not user_input.strip():
        return "", history
    answer = generate_response(user_input)
    history.append((user_input, answer))
    return "", history

# Example questions
examples = [
    "Why is my trading balance not updated even after adding funds successfully?",
    "Can I place an order when markets are closed (AMO Order)?",
    "What are brokerage charges?"
]

# Gradio UI
with gr.Blocks(title="📘 AngelOne Support Chatbot") as demo:
    gr.Markdown("## 💬 Chat with AngelOne Support Docs")

    chatbot_display = gr.Chatbot(label="AngelOne Docs Chat")
    user_input = gr.Textbox(placeholder="Ask a question about AngelOne support...", lines=2, show_label=False)
    send_btn = gr.Button("Send")
    state = gr.State([])

    # Example buttons
    with gr.Row():
        for q in examples:
            gr.Button(q).click(
                fn=chatbot,
                inputs=[gr.Textbox(value=q, visible=False), state],
                outputs=[user_input, chatbot_display]
            )

    # Send button or Enter key
    send_btn.click(fn=chatbot, inputs=[user_input, state], outputs=[user_input, chatbot_display])
    user_input.submit(fn=chatbot, inputs=[user_input, state], outputs=[user_input, chatbot_display])

# For Render / cloud hosting
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
