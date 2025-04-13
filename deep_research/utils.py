from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

def init_llm(
        provider: Literal["openai", "anthropic", "google", "ollama"],
        model: str,
        temperature: float = 0.5,
):
    """
    Initialize and return a language model chat interface based on the specified provider.

    This function creates a chat interface for different LLM providers including OpenAI, 
    Anthropic, Google, and Ollama. It handles API key validation and configuration for
    each provider.

    Args:
        provider: The LLM provider to use. Must be one of "openai", "anthropic", "google", or "ollama".
        model: The specific model name/identifier to use with the chosen provider.
        temperature: Controls randomness in the model's output. Higher values (e.g. 0.8) make the output
                    more random, while lower values (e.g. 0.2) make it more deterministic. Defaults to 0.5.

    Returns:
        A configured chat interface for the specified provider and model.

    Raises:
        ValueError: If the required API key environment variable is not set for the chosen provider
                   (except for Ollama which runs locally).
    """
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please set it in your environment variables.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=os.environ["GOOGLE_API_KEY"])
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)