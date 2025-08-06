from fastapi import FastAPI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel

app = FastAPI()

# Set the downloaded model that has been quantized according to the following documentation
# https://docs.vllm.ai/en/v0.7.0/features/quantization/auto_awq.html
model_id = "/home/jake/git/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-awq"

# Load a smaller model for demonstration purposes
engine_args = AsyncEngineArgs(
    model=model_id,
    gpu_memory_utilization=0.9,
    max_model_len=32168,
    quantization="AWQ",
    dtype="float16")
engine = AsyncLLMEngine.from_engine_args(engine_args)

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: str

@app.post("/generate")
async def generate(request: PromptRequest):
    # Set the stop parameter in SamplingParams to stop generation after the first answer:
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, stop=["\nUser:"])
    request_id = random_uuid()
    full_prompt = (
        f"You are a helpful assistant.\n"            
        f"{request.system_prompt}\n"
        f"User: {request.prompt}\n"
        f"Assistant:"
    )
    results_generator = engine.generate(full_prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        return {"text": ""}

    return {"text": final_output.outputs[0].text}
