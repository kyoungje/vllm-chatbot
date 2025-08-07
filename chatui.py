import gradio as gr
import httpx

# Custom CSS for chat bubbles
custom_css = """
.gradio-container {
    background-color: #f0f2f5;
}
.bubble-full-width {
    border: 1px solid #e0e0e0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    margin: 5px 0 !important;
}
.user {
    background-color: #dcf8c6 !important;
    border-radius: 15px 15px 0 15px !important;
    align-self: flex-end;
}
.bot {
    background-color: #ffffff !important;
    border-radius: 15px 15px 15px 0 !important;
    align-self: flex-start;
}
"""

async def get_bot_response(message, history):
    system_prompt = "Please keep your answers concise and to the point."

    # Convert Gradio's history to the backend's expected format
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg is not None:
            formatted_history.append({"role": "assistant", "content": bot_msg})

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8080/generate",
            json={"prompt": message,
                  "system_prompt": system_prompt,
                  "history": formatted_history
            },
            timeout=180.0  # Set a longer timeout
        )
    response.raise_for_status()
    return response.json()["text"]

with gr.Blocks(theme=gr.themes.Ocean(), css=custom_css, fill_height=True) as demo:
    chatbot_component = gr.Chatbot(
        scale=10,
        bubble_full_width=False,
        avatar_images=(
            "images/magnifying-glass.png",  # User avatar
            "images/robot.jpg"   # Bot avatar
        ),
        layout="bubble"
    )
    
    gr.ChatInterface(
        fn=get_bot_response,
        chatbot=chatbot_component,
        title="vLLM Chatbot",
        description="Ask any question to the chatbot.",
        examples=["What is WMX?", "Tell me about Movensys company", "What's software motion controller?"],
        fill_height=True
    )

demo.launch()
