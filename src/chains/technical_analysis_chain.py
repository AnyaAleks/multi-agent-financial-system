"""
Цепочка технического анализа
"""
from typing import Dict, Any, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalAnalysisSchema(BaseModel):
    """Схема для технического анализа"""
    overall_trend: str = Field(
        description="Overall market trend",
        pattern="^(BULLISH|BEARISH|SIDEWAYS|UNCLEAR)$"
    )
    trend_strength: float = Field(
        description="Strength of the trend from 0 to 1",
        ge=0,
        le=1
    )
    key_support_level: float = Field(
        description="Key support price level",
        ge=0
    )
    key_resistance_level: float = Field(
        description="Key resistance price level",
        ge=0
    )
    momentum_direction: str = Field(
        description="Current momentum direction",
        pattern="^(UP|DOWN|NEUTRAL)$"
    )
    volatility_level: str = Field(
        description="Current volatility level",
        pattern="^(LOW|MODERATE|HIGH|EXTREME)$"
    )
    market_phase: str = Field(
        description="Current market phase",
        pattern="^(ACCUMULATION|MARKUP|DISTRIBUTION|MARKDOWN)$"
    )
    risk_reward_ratio: float = Field(
        description="Estimated risk/reward ratio",
        ge=0
    )
    confidence_score: float = Field(
        description="Confidence in analysis from 0 to 1",
        ge=0,
        le=1
    )
    trading_signals: List[str] = Field(
        description="List of trading signals identified"
    )
    pattern_recognition: List[str] = Field(
        description="List of chart patterns recognized"
    )
    summary: str = Field(
        description="Summary of technical analysis"
    )

    @validator('key_resistance_level')
    def resistance_greater_than_support(cls, v, values):
        if 'key_support_level' in values and v <= values['key_support_level']:
            raise ValueError('Resistance level must be greater than support level')
        return v


class TechnicalAnalysisChain:
    """Цепочка технического анализа"""

    def __init__(self, llm):
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=TechnicalAnalysisSchema)
        self.chain = self._create_chain()

    def _create_chain(self) -> LLMChain:
        """Создание цепочки LangChain"""

        template = """
        Вы - старший технический аналитик с 20-летним опытом. 
        Проанализируйте следующие технические индикаторы и дайте комплексную оценку.

        Технические данные:
        {technical_data}

        Инструкции по анализу:
        1. Определите общий тренд (BULLISH/BEARISH/SIDEWAYS)
        2. Оцените силу тренда (0-1)
        3. Определите ключевые уровни поддержки и сопротивления
        4. Оцените направление импульса
        5. Определите уровень волатильности
        6. Идентифицируйте фазу рынка
        7. Рассчитайте соотношение риск/вознаграждение
        8. Перечислите торговые сигналы
        9. Распознайте графические паттерны
        10. Дайте итоговую оценку с уровнем уверенности

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["technical_data"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )

        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser
        )

    def run(self, technical_data: Dict[str, Any]) -> TechnicalAnalysisSchema:
        """Запуск цепочки технического анализа"""
        try:
            # Подготовка данных для анализа
            input_text = self._prepare_technical_data(technical_data)

            # Выполнение цепочки
            result = self.chain.run(technical_data=input_text)

            # Валидация результата
            validated_result = self._validate_analysis(result, technical_data)

            logger.info(f"TechnicalAnalysisChain: Анализ завершен")
            return validated_result

        except Exception as e:
            logger.error(f"TechnicalAnalysisChain: Ошибка: {e}")
            raise

    def _prepare_technical_data(self, technical_data: Dict[str, Any]) -> str:
        """Подготовка технических данных для анализа"""
        indicators = technical_data.get("indicators", {})

        analysis_text = "ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:\n\n"

        # RSI
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            analysis_text += f"RSI: {rsi.get('current', 'N/A')}\n"
            analysis_text += f"Уровень: {'Перекупленность' if rsi.get('overbought') else 'Перепроданность' if rsi.get('oversold') else 'Нейтральный'}\n\n"

        # MACD
        if "macd" in indicators:
            macd = indicators["macd"]
            analysis_text += f"MACD: {macd.get('macd', 'N/A')}\n"
            analysis_text += f"Signal: {macd.get('signal', 'N/A')}\n"
            analysis_text += f"Гистограмма: {macd.get('histogram', 'N/A')}\n"
            analysis_text += f"Пересечение: {macd.get('crossover', 'N/A')}\n\n"

        # Скользящие средние
        if "moving_averages" in indicators:
            ma = indicators["moving_averages"]
            analysis_text += f"SMA 20: {ma.get('sma_20', 'N/A')}\n"
            analysis_text += f"SMA 50: {ma.get('sma_50', 'N/A')}\n"
            analysis_text += f"SMA 200: {ma.get('sma_200', 'N/A')}\n"

            if ma.get("golden_cross"):
                analysis_text += "Золотой крест: ДА\n"
            elif ma.get("death_cross"):
                analysis_text += "Крест смерти: ДА\n"

            analysis_text += "\n"

        # Полосы Боллинджера
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            analysis_text += f"Верхняя полоса: {bb.get('upper', 'N/A')}\n"
            analysis_text += f"Средняя полоса: {bb.get('middle', 'N/A')}\n"
            analysis_text += f"Нижняя полоса: {bb.get('lower', 'N/A')}\n"
            if bb.get("percent_b") is not None:
                analysis_text += f"%B: {bb.get('percent_b'):.2f}\n"
            analysis_text += "\n"

        # Графические паттерны
        patterns = technical_data.get("patterns", [])
        if patterns:
            analysis_text += f"Графические паттерны: {', '.join(patterns)}\n\n"

        # Торговые сигналы
        signals = technical_data.get("signals", [])
        if signals:
            analysis_text += "ТОРГОВЫЕ СИГНАЛЫ:\n"
            for signal in signals[:5]:  # Ограничиваем 5 сигналами
                analysis_text += f"- {signal.get('type', 'N/A')} ({signal.get('indicator', 'N/A')}): {signal.get('description', '')}\n"
            analysis_text += "\n"

        # Ценовые данные для контекста
        if "price_data" in technical_data:
            price_data = technical_data["price_data"]
            if isinstance(price_data, list) and len(price_data) > 0:
                current = price_data[-1]
                analysis_text += f"Текущая цена: {current.get('close', 'N/A')}\n"

                if len(price_data) >= 2:
                    prev = price_data[-2]
                    change = ((current.get('close', 0) - prev.get('close', 0)) / prev.get('close', 0)) * 100
                    analysis_text += f"Изменение за день: {change:.2f}%\n"

        return analysis_text

    def _validate_analysis(self, analysis: TechnicalAnalysisSchema,
                           technical_data: Dict[str, Any]) -> TechnicalAnalysisSchema:
        """Валидация результатов анализа"""
        # Проверка согласованности данных
        indicators = technical_data.get("indicators", {})

        # Проверка RSI и тренда
        if "rsi" in indicators:
            rsi = indicators["rsi"].get("current")
            if rsi:
                if rsi > 70 and analysis.overall_trend == "BULLISH":
                    logger.warning("Предупреждение: RSI показывает перекупленность при бычьем тренде")
                elif rsi < 30 and analysis.overall_trend == "BEARISH":
                    logger.warning("Предупреждение: RSI показывает перепроданность при медвежьем тренде")

        # Проверка MACD
        if "macd" in indicators:
            macd_crossover = indicators["macd"].get("crossover")
            if macd_crossover == "bullish" and analysis.overall_trend == "BEARISH":
                logger.warning("Предупреждение: Бычье пересечение MACD при медвежьем тренде")
            elif macd_crossover == "bearish" and analysis.overall_trend == "BULLISH":
                logger.warning("Предупреждение: Медвежье пересечение MACD при бычьем тренде")

        # Корректировка уверенности на основе согласованности
        confidence = analysis.confidence_score

        # Проверка согласованности сигналов
        signals = technical_data.get("signals", [])
        buy_signals = sum(1 for s in signals if s.get("type") == "BUY")
        sell_signals = sum(1 for s in signals if s.get("type") == "SELL")

        if buy_signals > sell_signals and analysis.overall_trend == "BEARISH":
            confidence *= 0.8  # Уменьшаем уверенность при противоречии
        elif sell_signals > buy_signals and analysis.overall_trend == "BULLISH":
            confidence *= 0.8

        # Обновление уверенности
        analysis.confidence_score = min(1.0, confidence)

        return analysis

    def calculate_advanced_metrics(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет расширенных метрик"""
        if not price_data or len(price_data) < 20:
            return {}

        df = pd.DataFrame(price_data)
        closes = df["close"].values

        metrics = {}

        # ATR (Average True Range) - мера волатильности
        try:
            high = df["high"].values
            low = df["low"].values
            prev_close = np.roll(closes, 1)
            prev_close[0] = closes[0]

            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)

            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr[-14:])  # 14-дневный ATR
            metrics["atr"] = float(atr)
            metrics["atr_percent"] = float((atr / closes[-1]) * 100)
        except:
            pass

        # ADX (Average Directional Index) - сила тренда
        try:
            import talib
            adx = talib.ADX(df["high"].values, df["low"].values, closes, timeperiod=14)
            metrics["adx"] = float(adx[-1]) if not np.isnan(adx[-1]) else None

            # Интерпретация ADX
            if metrics["adx"]:
                if metrics["adx"] > 25:
                    metrics["trend_strength_adx"] = "STRONG"
                elif metrics["adx"] > 20:
                    metrics["trend_strength_adx"] = "MODERATE"
                else:
                    metrics["trend_strength_adx"] = "WEAK"
        except:
            pass

        # OBV (On-Balance Volume) - объемный анализ
        try:
            volumes = df["volume"].values
            obv = np.zeros_like(closes)

            for i in range(1, len(closes)):
                if closes[i] > closes[i - 1]:
                    obv[i] = obv[i - 1] + volumes[i]
                elif closes[i] < closes[i - 1]:
                    obv[i] = obv[i - 1] - volumes[i]
                else:
                    obv[i] = obv[i - 1]

            metrics["obv"] = float(obv[-1])
            metrics["obv_trend"] = "UP" if obv[-1] > obv[-5] else "DOWN"
        except:
            pass

        # Статистические метрики
        metrics["volatility_20d"] = float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100)
        metrics["sharpe_ratio_20d"] = self._calculate_sharpe_ratio(closes[-20:])

        # Уровни поддержки и сопротивления
        metrics["support_resistance"] = self._find_support_resistance(df)

        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) < 2:
            return 0.0

        daily_returns = np.diff(returns) / returns[:-1]
        excess_returns = daily_returns - (risk_free_rate / 252)  # Ежедневная безрисковая ставка

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Поиск уровней поддержки и сопротивления"""
        closes = df["close"].values

        if len(closes) < window * 2:
            return {"support": [], "resistance": []}

        support_levels = []
        resistance_levels = []

        # Поиск локальных минимумов и максимумов
        for i in range(window, len(closes) - window):
            local_window = closes[i - window:i + window]
            current_price = closes[i]

            # Проверка на локальный минимум (поддержка)
            if current_price == np.min(local_window):
                support_levels.append(float(current_price))

            # Проверка на локальный максимум (сопротивление)
            if current_price == np.max(local_window):
                resistance_levels.append(float(current_price))

        # Группировка близких уровней
        def group_levels(levels, tolerance=0.02):
            if not levels:
                return []

            levels = sorted(levels)
            grouped = []
            current_group = [levels[0]]

            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                    current_group.append(level)
                else:
                    grouped.append(np.mean(current_group))
                    current_group = [level]

            if current_group:
                grouped.append(np.mean(current_group))

            return grouped

        support = group_levels(support_levels)[-3:]  # Последние 3 уровня поддержки
        resistance = group_levels(resistance_levels)[-3:]  # Последние 3 уровня сопротивления

        return {
            "support": support,
            "resistance": resistance
        }