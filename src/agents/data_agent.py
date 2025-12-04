"""
Агент сбора данных
"""
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from src.agents.base_agent import BaseFinancialAgent
from src.mcp_servers.mcp_client import MCPClient
from src.utils.data_validation import DataValidator
from src.memory.short_term_memory import ShortTermMemory

from config.settings import settings


class DataAgent(BaseFinancialAgent):
    """Агент сбора и подготовки финансовых данных"""

    def __init__(self):
        tools = self._initialize_tools()

        super().__init__(
            role="Senior Financial Data Analyst",
            goal="Collect, clean, and prepare high-quality financial data for analysis",
            backstory=(
                "With over 10 years of experience at Bloomberg and Refinitiv, "
                "you specialize in extracting and processing financial data from "
                "multiple sources. You're meticulous about data quality and "
                "consistency, ensuring every dataset meets investment-grade standards."
            ),
            tools=tools,
            verbose=True
        )

        self.mcp_client = MCPClient(
            host=settings.mcp_financial_host,
            port=settings.mcp_financial_port
        )
        self.validator = DataValidator()
        self.memory = ShortTermMemory()

    def _initialize_tools(self) -> List:
        """Инициализация инструментов агента"""
        # Здесь будут инструменты LangChain
        return []

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение задачи сбора данных

        Args:
            task_input: {
                "ticker": "AAPL",
                "timeframe": "1y",
                "data_types": ["prices", "news", "fundamentals"]
            }
        """
        try:
            logger.info(f"DataAgent: Начинаю сбор данных для {task_input.get('ticker')}")

            # 1. Сбор данных через MCP
            collected_data = self._collect_data(task_input)

            # 2. Очистка и валидация
            cleaned_data = self._clean_and_validate(collected_data)

            # 3. Сохранение в память
            memory_key = f"data_{task_input['ticker']}_{datetime.now().timestamp()}"
            self.memory.store(memory_key, cleaned_data)

            # 4. Подготовка результата
            result = {
                "status": "success",
                "ticker": task_input["ticker"],
                "data_summary": self._create_data_summary(cleaned_data),
                "quality_score": self.validator.calculate_quality_score(cleaned_data),
                "memory_key": memory_key,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"DataAgent: Сбор данных завершен для {task_input['ticker']}")
            return result

        except Exception as e:
            logger.error(f"DataAgent: Ошибка при сборе данных: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _collect_data(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Сбор данных из различных источников"""
        ticker = task_input["ticker"]
        data_types = task_input.get("data_types", ["prices"])

        collected = {}

        for data_type in data_types:
            if data_type == "prices":
                # Получение ценовых данных через MCP
                prices = self.mcp_client.get_ohlcv(
                    symbol=ticker,
                    interval="1d",
                    period=task_input.get("timeframe", "1y")
                )
                collected["prices"] = prices

            elif data_type == "news":
                # Получение новостей через MCP
                news = self.mcp_client.get_news(
                    symbol=ticker,
                    days=30
                )
                collected["news"] = news

            elif data_type == "fundamentals":
                # Получение фундаментальных показателей
                fundamentals = self.mcp_client.get_fundamentals(ticker)
                collected["fundamentals"] = fundamentals

        return collected

    def _clean_and_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Очистка и валидация данных"""
        cleaned = {}

        if "prices" in data:
            df = pd.DataFrame(data["prices"])
            # Обработка пропущенных значений
            df = df.ffill().bfill()
            # Валидация схемы
            self.validator.validate_price_data(df)
            cleaned["prices"] = df.to_dict("records")

        if "news" in data:
            cleaned["news"] = self.validator.validate_news_data(data["news"])

        if "fundamentals" in data:
            cleaned["fundamentals"] = data["fundamentals"]

        return cleaned

    def _create_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Создание сводки по данным"""
        summary = {
            "data_points": 0,
            "time_period": {},
            "sources": []
        }

        if "prices" in data:
            prices = data["prices"]
            summary["data_points"] += len(prices)
            if prices:
                summary["time_period"]["start"] = prices[0].get("date")
                summary["time_period"]["end"] = prices[-1].get("date")
            summary["sources"].append("yahoo_finance")

        if "news" in data:
            summary["data_points"] += len(data["news"])
            summary["sources"].append("newsapi")

        return summary