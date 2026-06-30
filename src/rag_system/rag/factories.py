from langchain_openai import ChatOpenAI
from rag_system.core.config.settings import get_settings, LLMProvider

def build_llm():
    settings = get_settings()
    provider = settings.llm_provider

    # ── Ollama Provider ───────────────────────────────────────────────────────
    if provider == LLMProvider.OLLAMA:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            # Fallback if community import path changes
            from langchain_ollama import ChatOllama

        base_url = str(settings.ollama_base_url) if settings.ollama_base_url else "http://localhost:11434"
        return ChatOllama(
            base_url=base_url,
            model=settings.ollama_model,
            temperature=0,
        )

    # ── Anthropic Provider ────────────────────────────────────────────────────
    elif provider == LLMProvider.ANTHROPIC:
        key = settings.anthropic_api_key
        if not key or "mock" in key or "placeholder" in key:
            return _build_fake_llm()
            
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            from langchain_community.chat_models import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=key,
            temperature=0,
        )

    # ── OpenAI / Default Provider ─────────────────────────────────────────────
    else:
        key = settings.openai_api_key
        
        # Check if we are running with a placeholder / mock key and fall back to local fake LLM
        if not key or key.startswith("sk-abcdef") or "mock" in key or "YOUR_" in key:
            return _build_fake_llm()
            
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            streaming=True,
            api_key=key,
        )

def _build_fake_llm():
    try:
        from langchain_core.language_models.fake import FakeListLLM
    except ImportError:
        from langchain_community.llms.fake import FakeListLLM
        
    return FakeListLLM(
        responses=[
            "This is a simulated response from the RAG system. "
            "Because a mock/placeholder API key was detected, the system has processed your "
            "retrieved document context offline and generated this local simulation."
        ]
    )
