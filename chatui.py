import gradio as gr
import httpx

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

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.ChatInterface(
        fn=get_bot_response, 
        title="vLLM Chatbot",
        description="Ask any question to the chatbot.",
        examples=["What is WMX?", "Tell me about Movensys company", "what's soft motion controller?"],
        fill_height=True
    )

demo.launch()
