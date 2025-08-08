# vllm-chatbot
vllm-chatbot demo for serving llama model
gradio based chat UI

## Set-up
First, recommend creating a virtual environment.
To activate the virtual environment, enter: `source .venv/bin/activate`

Make sure you install the new dependency:
```sh
python3 -m venv .venv

source .venv/bin/activate

uv pip install -r requirements.txt
```

### vLLM instalation
Refer to this [official document](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#nvidia-cuda).

## Run

Run the backend server in a terminal:

```sh
uvicorn server:app --host 0.0.0.0 --port 8080
```
then, run the frontend chat application in a separate terminal:
```sh
python3 chatui.py
```
You can then access the chatbot interface in your web browser at the URL provided by Gradio (usually http://127.0.0.1:7860).