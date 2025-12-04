"""
Конфигурация системы
"""
import os
from typing import Dict, Any

class Settings:
    """Настройки системы"""

    # AI Provider
    AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

    # MCP Servers
    MCP_FINANCIAL_HOST = os.getenv("MCP_FINANCIAL_HOST", "localhost")
    MCP_FINANCIAL_PORT = int(os.getenv("MCP_FINANCIAL_PORT", "8000"))

    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))

    # Neo4j
    NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Конфигурация LLM"""
        return {
            "provider": self.AI_PROVIDER,
            "model": self.OLLAMA_MODEL,
            "temperature": 0.1,
            "base_url": self.OLLAMA_BASE_URL
        }

    @property
    def ai_provider(self) -> str:
        """AI провайдер"""
        return self.AI_PROVIDER

    @property
    def ollama_base_url(self) -> str:
        """URL Ollama"""
        return self.OLLAMA_BASE_URL

    @property
    def ollama_model(self) -> str:
        """Модель Ollama"""
        return self.OLLAMA_MODEL

    @property
    def environment(self) -> str:
        """Окружение"""
        return self.ENVIRONMENT

settings = Settings()