"""
Цепочка анализа финансовых данных
"""
from typing import Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialDataSchema(BaseModel):
    """Схема для валидации финансовых данных"""
    ticker: str = Field(description="Stock ticker symbol")
    time_period: str = Field(description="Analysis time period")
    data_quality_score: float = Field(
        description="Data quality score from 0 to 100",
        ge=0,
        le=100
    )
    missing_values_percentage: float = Field(
        description="Percentage of missing values",
        ge=0,
        le=100
    )
    outliers_detected: int = Field(
        description="Number of statistical outliers detected",
        ge=0
    )
    validation_report: str = Field(description="Detailed validation report")
    recommendation: str = Field(
        description="Recommendation for data usage",
        pattern="^(USE_AS_IS|NEEDS_CLEANING|DO_NOT_USE)$"
    )


class FinancialDataAnalysisChain:
    """Цепочка анализа финансовых данных"""

    def __init__(self, llm):
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=FinancialDataSchema)
        self.chain = self._create_chain()

    def _create_chain(self) -> LLMChain:
        """Создание цепочки LangChain"""

        template = """
        Вы старший финансовый аналитик. Проанализируйте предоставленные финансовые данные.

        Входные данные:
        {input_data}

        Требования к анализу:
        1. Оцените качество данных (0-100)
        2. Определите процент пропущенных значений
        3. Выявите статистические выбросы
        4. Проверьте соответствие схеме OHLCV
        5. Дайте рекомендацию по использованию

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["input_data"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )

        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser
        )

    def run(self, data: Dict[str, Any]) -> FinancialDataSchema:
        """Запуск цепочки анализа"""
        try:
            # Преобразование данных в текстовый формат
            input_text = self._prepare_input(data)

            # Выполнение цепочки
            result = self.chain.run(input_data=input_text)

            logger.info(f"FinancialDataAnalysisChain: Анализ завершен")
            return result

        except Exception as e:
            logger.error(f"FinancialDataAnalysisChain: Ошибка: {e}")
            raise

    def _prepare_input(self, data: Dict[str, Any]) -> str:
        """Подготовка входных данных"""
        input_parts = []

        if "prices" in data:
            prices = data["prices"]
            if isinstance(prices, list) and len(prices) > 0:
                sample = prices[:5]  # Первые 5 записей для анализа
                input_parts.append(f"Ценовые данные (образец): {sample}")
                input_parts.append(f"Всего записей: {len(prices)}")

        if "news" in data:
            news_count = len(data["news"])
            input_parts.append(f"Новостных статей: {news_count}")

        if "fundamentals" in data:
            input_parts.append(f"Фундаментальные показатели: {data['fundamentals']}")

        return "\n".join(input_parts)

    def calculate_quality_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет метрик качества данных"""
        metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "accuracy": 0.0
        }

        if "prices" in data:
            df = pd.DataFrame(data["prices"])

            # Полнота данных
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            metrics["completeness"] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

            # Консистентность (проверка OHLCV)
            required_columns = ["open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_columns):
                metrics["consistency"] = 1.0

            # Своевременность
            if "date" in df.columns:
                latest_date = pd.to_datetime(df["date"].max())
                days_old = (pd.Timestamp.now() - latest_date).days
                metrics["timeliness"] = max(0, 1 - (days_old / 30))  # 30 дней - порог

        return metrics