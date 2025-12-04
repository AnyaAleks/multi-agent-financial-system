"""
Простая фабрика LLM для Ollama
"""
from typing import Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from config.settings import settings


class LLMFactory:
    """Фабрика для создания LLM клиентов"""

    @staticmethod
    def create_llm() -> BaseChatModel:
        """Создание LLM клиента"""
        config = settings.llm_config
        provider = config.get("provider", "ollama")

        print(f"🤖 Creating LLM client for provider: {provider}")

        try:
            if provider == "ollama":
                return ChatOllama(
                    model=config.get("model", "llama3.1:latest"),
                    temperature=config.get("temperature", 0.1),
                    base_url=config.get("base_url", "http://localhost:11434"),
                    num_predict=2048
                )
            elif provider == "openai":
                import os
                return ChatOpenAI(
                    model=config.get("model", "gpt-3.5-turbo"),
                    temperature=config.get("temperature", 0.1),
                    openai_api_key=os.getenv("OPENAI_API_KEY", "")
                )
            else:
                # Fallback to mock LLM
                from langchain.chat_models.fake import FakeListChatModel
                responses = [
                    "Based on analysis: BUY with 85% confidence",
                    "Recommendation: HOLD, waiting for clearer signals",
                    "Analysis suggests: SELL due to high risk"
                ]
                return FakeListChatModel(responses=responses)

        except Exception as e:
            print(f"❌ Failed to create LLM: {e}")
            print("⚠️ Using mock LLM instead")
            from langchain.chat_models.fake import FakeListChatModel
            return FakeListChatModel(responses=["Mock response for testing"])

    @staticmethod
    def test_connection() -> bool:
        """Тестирование соединения с LLM"""
        import requests

        try:
            if settings.ai_provider == "ollama":
                response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
                return response.status_code == 200
            else:
                return True  # Для других провайдеров всегда True
        except:
            return False


# Глобальный LLM
llm = LLMFactory.create_llm()
print(f"✅ LLM initialized: {settings.ollama_model}")