import gradio as gr
import httpx

async def get_bot_response(message, history):
    system_prompt = "Please keep your answers concise and to the point."  
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8080/generate", 
            json={"prompt": message, "system_prompt": system_prompt},
            timeout=300.0  # Set a longer timeout
        )
    response.raise_for_status()
    return response.json()["text"]

demo = gr.ChatInterface(
    fn=get_bot_response, 
    title="vLLM Chatbot",
    description="Ask any question to the chatbot."
)

demo.launch()
