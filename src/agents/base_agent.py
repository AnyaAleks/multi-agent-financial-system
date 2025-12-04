"""
Базовый класс для всех агентов
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from crewai import Agent
from langchain.tools import BaseTool

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFinancialAgent(ABC):
    """Базовый агент финансового анализа"""

    def __init__(
            self,
            role: str,
            goal: str,
            backstory: str,
            tools: Optional[list] = None,
            verbose: bool = True,
            memory: bool = True
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose
        self.memory = memory
        self.agent: Optional[Agent] = None
        self.initialize_agent()

    def initialize_agent(self):
        """Инициализация агента CrewAI"""
        try:
            self.agent = Agent(
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                tools=self.tools,
                verbose=self.verbose,
                memory=self.memory,
                allow_delegation=False
            )
            logger.info(f"Агент {self.role} инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации агента {self.role}: {e}")
            raise

    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение задачи агентом"""
        pass

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Валидация выходных данных"""
        # Базовая валидация
        required_fields = ["status", "data", "timestamp"]
        return all(field in output for field in required_fields)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Метрики производительности агента"""
        return {
            "agent_role": self.role,
            "tasks_completed": 0,  # Будет обновляться
            "average_execution_time": 0,
            "success_rate": 1.0,
            "last_execution": None
        }