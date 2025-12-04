"""
Менеджер-агент для оркестрации workflow
"""
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
from enum import Enum

from src.agents.base_agent import BaseFinancialAgent
from src.agents.data_agent import DataAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.report_agent import ReportAgent
from src.memory.short_term_memory import ShortTermMemory
from src.monitoring.health_monitor import HealthMonitor

from config.settings import settings


class TaskStatus(Enum):
    """Статусы выполнения задач"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Приоритеты задач"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ManagerAgent(BaseFinancialAgent):
    """Менеджер-агент для оркестрации workflow"""

    def __init__(self, data_agent: DataAgent, analysis_agent: AnalysisAgent,
                 report_agent: ReportAgent):
        super().__init__(
            role="Workflow Orchestrator",
            goal="Coordinate and manage the execution of financial analysis tasks",
            backstory=(
                "Former project manager at a top-tier investment bank with "
                "expertise in process optimization and resource allocation. "
                "You ensure that every analysis task is executed efficiently, "
                "on time, and with the highest quality standards."
            ),
            tools=[],
            verbose=True
        )

        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.report_agent = report_agent

        self.memory = ShortTermMemory()
        self.monitor = HealthMonitor()
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []

        # Статистика
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }

    async def execute_workflow(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение полного workflow анализа

        Args:
            task_input: {
                "ticker": "AAPL",
                "timeframe": "1y",
                "analysis_type": "basic|standard|deep",
                "report_type": "pdf|dashboard|executive_summary",
                "priority": "low|normal|high|critical"
            }
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_input['ticker']}"

        task = {
            "task_id": task_id,
            "input": task_input,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "results": {},
            "errors": [],
            "retry_count": 0,
            "max_retries": 3
        }

        self.active_tasks[task_id] = task
        self.task_history.append(task)
        self.stats["total_tasks"] += 1

        logger.info(f"ManagerAgent: Начинаю выполнение задачи {task_id}")

        try:
            task["started_at"] = datetime.now().isoformat()
            task["status"] = TaskStatus.RUNNING.value

            # 1. Сбор данных
            data_result = await self._execute_data_collection(task_input, task_id)
            if data_result["status"] != "success":
                raise Exception(f"Data collection failed: {data_result.get('error')}")

            # 2. Анализ данных
            analysis_result = await self._execute_analysis(
                data_result, task_input, task_id
            )
            if analysis_result["status"] != "success":
                raise Exception(f"Analysis failed: {analysis_result.get('error')}")

            # 3. Генерация отчета
            report_result = await self._execute_report_generation(
                analysis_result, task_input, task_id
            )
            if report_result["status"] != "success":
                raise Exception(f"Report generation failed: {report_result.get('error')}")

            # 4. Сборка финального результата
            final_result = self._compile_final_result(
                data_result, analysis_result, report_result, task_input
            )

            task["status"] = TaskStatus.SUCCESS.value
            task["completed_at"] = datetime.now().isoformat()
            task["results"] = final_result

            self.stats["successful_tasks"] += 1

            # Расчет времени выполнения
            exec_time = self._calculate_execution_time(task)
            self.stats["total_execution_time"] += exec_time
            self.stats["average_execution_time"] = (
                    self.stats["total_execution_time"] / self.stats["successful_tasks"]
            )

            logger.info(f"ManagerAgent: Задача {task_id} выполнена успешно за {exec_time:.2f} сек")

            return final_result

        except Exception as e:
            logger.error(f"ManagerAgent: Ошибка выполнения задачи {task_id}: {e}")

            task["status"] = TaskStatus.FAILED.value
            task["completed_at"] = datetime.now().isoformat()
            task["errors"].append(str(e))

            self.stats["failed_tasks"] += 1

            # Попытка повторного выполнения
            if task["retry_count"] < task["max_retries"]:
                logger.info(f"ManagerAgent: Повторная попытка задачи {task_id}")
                task["retry_count"] += 1
                task["status"] = TaskStatus.RETRYING.value

                # Добавляем задачу в очередь для повторного выполнения
                await self.task_queue.put(task)

                return {
                    "status": "retrying",
                    "task_id": task_id,
                    "message": f"Task failed, retrying ({task['retry_count']}/{task['max_retries']})",
                    "error": str(e)
                }

            # Возвращаем ошибку
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e),
                "task_details": task
            }

    async def _execute_data_collection(self, task_input: Dict[str, Any],
                                       task_id: str) -> Dict[str, Any]:
        """Выполнение сбора данных"""
        logger.info(f"ManagerAgent: Запуск DataAgent для задачи {task_id}")

        # Подготовка задачи для DataAgent
        data_task = {
            "ticker": task_input["ticker"],
            "timeframe": task_input.get("timeframe", "1y"),
            "data_types": self._get_data_types_for_analysis(task_input.get("analysis_type", "standard"))
        }

        try:
            # Проверка здоровья агента
            if not self.monitor.check_agent_health("data_agent"):
                logger.warning("DataAgent нездоров, использование запасного варианта")
                return await self._fallback_data_collection(data_task)

            # Выполнение задачи
            result = await asyncio.to_thread(self.data_agent.execute_task, data_task)

            # Сохранение в память
            memory_key = f"{task_id}_data"
            self.memory.store(memory_key, {
                "task_id": task_id,
                "data": result,
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"ManagerAgent: Ошибка DataAgent: {e}")
            raise

    async def _execute_analysis(self, data_result: Dict[str, Any],
                                task_input: Dict[str, Any],
                                task_id: str) -> Dict[str, Any]:
        """Выполнение анализа данных"""
        logger.info(f"ManagerAgent: Запуск AnalysisAgent для задачи {task_id}")

        # Подготовка задачи для AnalysisAgent
        analysis_task = {
            "data_key": f"{task_id}_data",
            "analysis_type": task_input.get("analysis_type", "standard"),
            "parameters": {
                "include_technical": True,
                "include_sentiment": True,
                "include_fundamental": task_input.get("analysis_type") == "deep",
                "risk_assessment_depth": self._get_risk_depth(task_input.get("analysis_type"))
            }
        }

        try:
            # Проверка здоровья агента
            if not self.monitor.check_agent_health("analysis_agent"):
                logger.warning("AnalysisAgent нездоров, использование запасного варианта")
                return await self._fallback_analysis(data_result, analysis_task)

            # Выполнение задачи
            result = await asyncio.to_thread(self.analysis_agent.execute_task, analysis_task)

            # Сохранение в память
            memory_key = f"{task_id}_analysis"
            self.memory.store(memory_key, {
                "task_id": task_id,
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"ManagerAgent: Ошибка AnalysisAgent: {e}")
            raise

    async def _execute_report_generation(self, analysis_result: Dict[str, Any],
                                         task_input: Dict[str, Any],
                                         task_id: str) -> Dict[str, Any]:
        """Выполнение генерации отчета"""
        logger.info(f"ManagerAgent: Запуск ReportAgent для задачи {task_id}")

        # Подготовка задачи для ReportAgent
        report_task = {
            "analysis_key": f"{task_id}_analysis",
            "report_type": task_input.get("report_type", "executive_summary"),
            "output_format": self._get_output_format(task_input.get("report_type")),
            "customizations": {
                "include_charts": True,
                "include_metrics": True,
                "include_recommendations": True,
                "branding": {
                    "company_name": "AI Financial Analysis System",
                    "logo_url": None
                }
            }
        }

        try:
            # Проверка здоровья агента
            if not self.monitor.check_agent_health("report_agent"):
                logger.warning("ReportAgent нездоров, использование запасного варианта")
                return await self._fallback_report_generation(analysis_result, report_task)

            # Выполнение задачи
            result = await asyncio.to_thread(self.report_agent.execute_task, report_task)

            # Сохранение в память
            memory_key = f"{task_id}_report"
            self.memory.store(memory_key, {
                "task_id": task_id,
                "report": result,
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"ManagerAgent: Ошибка ReportAgent: {e}")
            raise

    def _compile_final_result(self, data_result: Dict[str, Any],
                              analysis_result: Dict[str, Any],
                              report_result: Dict[str, Any],
                              task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Сборка финального результата"""
        return {
            "status": "success",
            "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ticker": task_input["ticker"],
            "timeframe": task_input.get("timeframe", "1y"),
            "analysis_type": task_input.get("analysis_type", "standard"),
            "report_type": task_input.get("report_type", "executive_summary"),
            "summary": {
                "recommendation": analysis_result.get("recommendation", {}).get("action", "HOLD"),
                "confidence": analysis_result.get("confidence_score", 0.7),
                "risk_level": analysis_result.get("risk_assessment", {}).get("risk_level", "MEDIUM"),
                "market_outlook": analysis_result.get("insights", {}).get("market_outlook", "Neutral")
            },
            "data_quality": data_result.get("quality_score", 0),
            "analysis_results": {
                "technical": analysis_result.get("technical_analysis", {}),
                "sentiment": analysis_result.get("sentiment_analysis", {}),
                "insights": analysis_result.get("insights", {}),
                "risk": analysis_result.get("risk_assessment", {})
            },
            "report": {
                "type": report_result.get("report_type", "unknown"),
                "path": report_result.get("report_path"),
                "url": report_result.get("report_url"),
                "format": report_result.get("format", "unknown")
            },
            "timestamps": {
                "data_collection": data_result.get("timestamp"),
                "analysis": analysis_result.get("timestamp"),
                "report_generation": report_result.get("timestamp"),
                "workflow_completed": datetime.now().isoformat()
            },
            "metadata": {
                "version": "1.0",
                "system_id": "multi-agent-financial-system"
            }
        }

    async def _fallback_data_collection(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Запасной вариант сбора данных"""
        logger.warning("Использую запасной вариант сбора данных")

        # Упрощенный сбор данных
        try:
            import yfinance as yf

            ticker = yf.Ticker(task["ticker"])
            hist = ticker.history(period=task.get("timeframe", "1y"))

            data = []
            for idx, row in hist.iterrows():
                data.append({
                    "date": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })

            return {
                "status": "success",
                "ticker": task["ticker"],
                "data_type": "fallback",
                "prices": data,
                "quality_score": 0.7,  # Ниже качество
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Запасной сбор данных также не удался: {e}")
            raise

    async def _fallback_analysis(self, data_result: Dict[str, Any],
                                 analysis_task: Dict[str, Any]) -> Dict[str, Any]:
        """Запасной вариант анализа"""
        logger.warning("Использую запасной вариант анализа")

        # Упрощенный анализ
        try:
            if "prices" in data_result:
                prices = data_result["prices"]
                if isinstance(prices, list) and len(prices) > 0:
                    current_price = prices[-1]["close"]
                    first_price = prices[0]["close"]
                    price_change = ((current_price - first_price) / first_price) * 100

                    return {
                        "status": "success",
                        "analysis_type": "fallback",
                        "technical_analysis": {
                            "price_change": f"{price_change:.2f}%",
                            "current_price": current_price
                        },
                        "recommendation": {
                            "action": "BUY" if price_change > 0 else "SELL",
                            "strength": "WEAK",
                            "reasoning": "Based on simple price change analysis"
                        },
                        "confidence_score": 0.5,
                        "timestamp": datetime.now().isoformat()
                    }

            return {
                "status": "success",
                "analysis_type": "fallback",
                "message": "Minimal analysis completed",
                "confidence_score": 0.3,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Запасной анализ также не удался: {e}")
            raise

    async def _fallback_report_generation(self, analysis_result: Dict[str, Any],
                                          report_task: Dict[str, Any]) -> Dict[str, Any]:
        """Запасной вариант генерации отчета"""
        logger.warning("Использую запасной вариант генерации отчета")

        try:
            ticker = analysis_result.get("ticker", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/{ticker}_fallback_{timestamp}.txt"

            os.makedirs("reports", exist_ok=True)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Fallback Report for {ticker}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if "recommendation" in analysis_result:
                    rec = analysis_result["recommendation"]
                    f.write(f"Recommendation: {rec.get('action', 'N/A')}\n")
                    f.write(f"Reasoning: {rec.get('reasoning', 'N/A')}\n")

                f.write("\nNote: This is a fallback report generated due to system issues.\n")

            return {
                "status": "success",
                "report_type": "fallback",
                "report_path": filename,
                "format": "text",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Запасная генерация отчета также не удалась: {e}")
            raise

    def _get_data_types_for_analysis(self, analysis_type: str) -> List[str]:
        """Определение типов данных для анализа"""
        if analysis_type == "basic":
            return ["prices"]
        elif analysis_type == "standard":
            return ["prices", "news"]
        elif analysis_type == "deep":
            return ["prices", "news", "fundamentals"]
        else:
            return ["prices", "news"]

    def _get_risk_depth(self, analysis_type: str) -> str:
        """Определение глубины оценки рисков"""
        if analysis_type == "deep":
            return "detailed"
        elif analysis_type == "standard":
            return "moderate"
        else:
            return "basic"

    def _get_output_format(self, report_type: str) -> str:
        """Определение формата вывода"""
        if report_type == "pdf":
            return "pdf"
        elif report_type == "dashboard":
            return "html"
        else:
            return "text"

    def _calculate_execution_time(self, task: Dict[str, Any]) -> float:
        """Расчет времени выполнения задачи"""
        if task["started_at"] and task["completed_at"]:
            start = datetime.fromisoformat(task["started_at"])
            end = datetime.fromisoformat(task["completed_at"])
            return (end - start).total_seconds()
        return 0.0

    async def process_task_queue(self):
        """Обработка очереди задач"""
        logger.info("ManagerAgent: Запуск обработки очереди задач")

        while True:
            try:
                task = await self.task_queue.get()

                if task["status"] == TaskStatus.RETRYING.value:
                    logger.info(f"Повторное выполнение задачи {task['task_id']}")
                    await self.execute_workflow(task["input"])

                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка обработки очереди задач: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса задачи"""
        return self.active_tasks.get(task_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "total_history_tasks": len(self.task_history),
            "queue_size": self.task_queue.qsize(),
            "agents_health": {
                "data_agent": self.monitor.check_agent_health("data_agent"),
                "analysis_agent": self.monitor.check_agent_health("analysis_agent"),
                "report_agent": self.monitor.check_agent_health("report_agent")
            },
            "memory_usage": self.memory.get_usage_stats(),
            "timestamp": datetime.now().isoformat()
        }

    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Очистка завершенных задач"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        tasks_to_remove = []
        for task_id, task in self.active_tasks.items():
            if task["status"] in [TaskStatus.SUCCESS.value, TaskStatus.FAILED.value]:
                completed_at = task.get("completed_at")
                if completed_at:
                    try:
                        completed_time = datetime.fromisoformat(completed_at).timestamp()
                        if completed_time < cutoff_time:
                            tasks_to_remove.append(task_id)
                    except:
                        pass

        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]

        logger.info(f"Очищено {len(tasks_to_remove)} завершенных задач")