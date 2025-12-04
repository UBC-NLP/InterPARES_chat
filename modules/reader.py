from huggingface_hub import InferenceClient
from modules.utils import getconfig
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

model_config = getconfig("model_params.cfg")



def inf_provider():
    """return the inf service provider"""
    provider = model_config.get('reader','INF_PROVIDER')
    
    # Create LangChain ChatOpenAI client for vLLM server
    client = ChatOpenAI(
        base_url="http://localhost:8000/v1",  # vLLM OpenAI API endpoint
        api_key="EMPTY",  # vLLM doesn't require real API key
        model=model_config.get('reader','INF_PROVIDER_MODEL'),
        temperature=0.1,
        max_tokens=int(model_config.get('reader','MAX_TOKENS')),
        streaming=True
    )
    
    print(f"getting {provider} client (vLLM LangChain-compatible)")
    return client



