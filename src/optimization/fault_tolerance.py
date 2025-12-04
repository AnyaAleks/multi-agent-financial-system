"""
Механизмы отказоустойчивости
"""
import time
import random
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from functools import wraps
import signal
import sys

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class CircuitState(Enum):
    """Состояния автомата"""
    CLOSED = "closed"  # Нормальная работа
    OPEN = "open"  # Сервис недоступен
    HALF_OPEN = "half_open"  # Пробное восстановление


class FailureType(Enum):
    """Типы сбоев"""
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitBreaker:
    """Автомат для предотвращения каскадных сбоев"""

    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 30,
                 half_open_max_calls: int = 3):

        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()

        # Статистика
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opens": 0,
            "circuit_closes": 0,
            "rejected_calls": 0
        }

        logger.info(f"CircuitBreaker {name}: Инициализирован")

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Выполнение функции с защитой автомата

        Args:
            func: Функция для выполнения
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения

        Raises:
            CircuitBreakerError: Если автомат открыт
            Exception: Ошибки выполнения функции
        """
        self.stats["total_calls"] += 1

        # Проверка состояния автомата
        if self.state == CircuitState.OPEN:
            # Проверка, не истекло ли время восстановления
            if self._should_try_recovery():
                self._transition_to_half_open()
            else:
                self.stats["rejected_calls"] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )

        elif self.state == CircuitState.HALF_OPEN:
            # Ограничение количества вызовов в HALF_OPEN состоянии
            if self.success_count >= self.half_open_max_calls:
                self._transition_to_closed()
            elif self.failure_count > 0:
                self._transition_to_open()

        try:
            # Выполнение функции
            result = func(*args, **kwargs)

            # Обработка успешного выполнения
            self._on_success()
            self.stats["successful_calls"] += 1

            return result

        except Exception as e:
            # Обработка ошибки
            failure_type = self._classify_error(e)
            self._on_failure(failure_type)
            self.stats["failed_calls"] += 1

            raise

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Асинхронное выполнение функции с защитой автомата

        Args:
            func: Асинхронная функция для выполнения
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения
        """
        self.stats["total_calls"] += 1

        # Проверка состояния автомата
        if self.state == CircuitState.OPEN:
            if self._should_try_recovery():
                self._transition_to_half_open()
            else:
                self.stats["rejected_calls"] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN (async). "
                    f"Last failure: {self.last_failure_time}"
                )

        try:
            # Асинхронное выполнение
            result = await func(*args, **kwargs)

            # Обработка успеха
            self._on_success()
            self.stats["successful_calls"] += 1

            return result

        except Exception as e:
            # Обработка ошибки
            failure_type = self._classify_error(e)
            self._on_failure(failure_type)
            self.stats["failed_calls"] += 1

            raise

    def _on_success(self):
        """Обработка успешного выполнения"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            # Если достигли лимита успешных вызовов, закрываем автомат
            if self.success_count >= self.half_open_max_calls:
                self._transition_to_closed()

        elif self.state == CircuitState.CLOSED:
            # Сброс счетчика ошибок при успешных вызовах
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, failure_type: FailureType):
        """Обработка ошибки выполнения"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        # Логирование ошибки
        logger.warning(
            f"CircuitBreaker {self.name}: Failure #{self.failure_count}. "
            f"Type: {failure_type.value}. State: {self.state.value}"
        )

        # Проверка порога сбоев
        if (self.state == CircuitState.CLOSED and
                self.failure_count >= self.failure_threshold):
            self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Любая ошибка в HALF_OPEN состоянии возвращает в OPEN
            self._transition_to_open()

    def _transition_to_open(self):
        """Переход в состояние OPEN"""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            self.stats["circuit_opens"] += 1

            logger.warning(f"CircuitBreaker {self.name}: Transitioned to OPEN state")

    def _transition_to_closed(self):
        """Переход в состояние CLOSED"""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_state_change = datetime.now()
            self.stats["circuit_closes"] += 1

            logger.info(f"CircuitBreaker {self.name}: Transitioned to CLOSED state")

    def _transition_to_half_open(self):
        """Переход в состояние HALF_OPEN"""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.failure_count = 0
            self.last_state_change = datetime.now()

            logger.info(f"CircuitBreaker {self.name}: Transitioned to HALF_OPEN state")

    def _should_try_recovery(self) -> bool:
        """Проверка возможности попытки восстановления"""
        if self.state == CircuitState.OPEN and self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            return time_since_failure.total_seconds() >= self.recovery_timeout
        return False

    def _classify_error(self, error: Exception) -> FailureType:
        """Классификация ошибки"""
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return FailureType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return FailureType.NETWORK_ERROR
        elif "unavailable" in error_str or "503" in error_str:
            return FailureType.SERVICE_UNAVAILABLE
        elif "validation" in error_str or "invalid" in error_str:
            return FailureType.VALIDATION_ERROR
        else:
            return FailureType.UNKNOWN_ERROR

    def get_state_info(self) -> Dict[str, Any]:
        """Получение информации о состоянии автомата"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
            "stats": self.stats,
            "thresholds": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls
            },
            "timestamp": datetime.now().isoformat()
        }

    def reset(self):
        """Сброс автомата в начальное состояние"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()

        logger.info(f"CircuitBreaker {self.name}: Reset to initial state")


class CircuitBreakerError(Exception):
    """Ошибка автомата"""
    pass


def circuit_breaker(name: str, **breaker_kwargs):
    """
    Декоратор для автоматического применения автомата

    Args:
        name: Имя автомата
        **breaker_kwargs: Параметры автомата

    Returns:
        Декорированная функция
    """
    # Глобальный реестр автоматов
    if not hasattr(circuit_breaker, '_breakers'):
        circuit_breaker._breakers = {}

    def decorator(func):
        # Создание или получение автомата
        if name not in circuit_breaker._breakers:
            circuit_breaker._breakers[name] = CircuitBreaker(name, **breaker_kwargs)

        breaker = circuit_breaker._breakers[name]

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.execute(func, *args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.execute_async(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


class RetryManager:
    """Менеджер повторных попыток"""

    def __init__(self,
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 30.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):

        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries_performed": 0,
            "total_retry_delay": 0.0
        }

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Выполнение функции с повторными попытками

        Args:
            func: Функция для выполнения
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения

        Raises:
            Exception: После исчерпания всех попыток
        """
        self.stats["total_calls"] += 1

        last_exception = None
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                if retry_count > 0:
                    # Задержка перед повторной попыткой
                    delay = self._calculate_delay(retry_count)
                    self.stats["total_retry_delay"] += delay

                    logger.info(
                        f"RetryManager: Retry #{retry_count} for {func.__name__}. "
                        f"Delay: {delay:.2f}s"
                    )

                    time.sleep(delay)

                # Выполнение функции
                result = func(*args, **kwargs)

                # Обновление статистики
                if retry_count > 0:
                    self.stats["retries_performed"] += retry_count

                self.stats["successful_calls"] += 1
                return result

            except Exception as e:
                last_exception = e
                retry_count += 1

                # Проверка, стоит ли повторять
                if not self._should_retry(e, retry_count):
                    break

        # Все попытки исчерпаны
        self.stats["failed_calls"] += 1

        logger.error(
            f"RetryManager: All {self.max_retries} retries failed for {func.__name__}. "
            f"Last error: {last_exception}"
        )

        raise last_exception

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Асинхронное выполнение с повторными попытками

        Args:
            func: Асинхронная функция
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения
        """
        self.stats["total_calls"] += 1

        last_exception = None
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                if retry_count > 0:
                    # Асинхронная задержка
                    delay = self._calculate_delay(retry_count)
                    self.stats["total_retry_delay"] += delay

                    logger.info(
                        f"RetryManager: Async retry #{retry_count} for {func.__name__}. "
                        f"Delay: {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

                # Асинхронное выполнение
                result = await func(*args, **kwargs)

                # Обновление статистики
                if retry_count > 0:
                    self.stats["retries_performed"] += retry_count

                self.stats["successful_calls"] += 1
                return result

            except Exception as e:
                last_exception = e
                retry_count += 1

                if not self._should_retry(e, retry_count):
                    break

        # Все попытки исчерпаны
        self.stats["failed_calls"] += 1

        logger.error(
            f"RetryManager: All {self.max_retries} async retries failed for {func.__name__}. "
            f"Last error: {last_exception}"
        )

        raise last_exception

    def _calculate_delay(self, retry_count: int) -> float:
        """Расчет задержки для повторной попытки"""
        # Экспоненциальная задержка
        delay = self.initial_delay * (self.backoff_factor ** (retry_count - 1))

        # Ограничение максимальной задержки
        delay = min(delay, self.max_delay)

        # Добавление случайности (jitter)
        if self.jitter:
            jitter = random.uniform(0.5, 1.5)
            delay *= jitter

        return delay

    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Определение, стоит ли повторять попытку"""
        # Не повторять, если превышено количество попыток
        if retry_count > self.max_retries:
            return False

        # Анализ типа ошибки
        error_str = str(error).lower()

        # Повторять для временных ошибок
        retryable_errors = [
            "timeout", "connection", "network", "temporarily",
            "busy", "overloaded", "503", "504", "502"
        ]

        for retryable in retryable_errors:
            if retryable in error_str:
                return True

        # Не повторять для ошибок валидации и бизнес-логики
        non_retryable_errors = [
            "validation", "invalid", "not found", "permission",
            "authorization", "authentication"
        ]

        for non_retryable in non_retryable_errors:
            if non_retryable in error_str:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        success_rate = (self.stats["successful_calls"] / self.stats["total_calls"] * 100
                        if self.stats["total_calls"] > 0 else 0)

        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "average_retries_per_call": (
                self.stats["retries_performed"] / self.stats["total_calls"]
                if self.stats["total_calls"] > 0 else 0
            ),
            "timestamp": datetime.now().isoformat()
        }


def retry(max_retries=3, **retry_kwargs):
    """
    Декоратор для автоматического применения повторных попыток

    Args:
        max_retries: Максимальное количество попыток
        **retry_kwargs: Параметры менеджера повторных попыток

    Returns:
        Декорированная функция
    """

    def decorator(func):
        retry_manager = RetryManager(max_retries=max_retries, **retry_kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_manager.execute(func, *args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.execute_async(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


class FallbackManager:
    """Менеджер запасных вариантов"""

    def __init__(self, primary_func: Callable[..., T],
                 fallback_funcs: List[Callable[..., T]]):
        self.primary_func = primary_func
        self.fallback_funcs = fallback_funcs
        self.current_func_index = 0

        self.stats = {
            "primary_success": 0,
            "fallback_success": 0,
            "total_failures": 0,
            "fallback_activations": 0
        }

    def execute(self, *args, **kwargs) -> T:
        """
        Выполнение с запасными вариантами

        Args:
            *args: Аргументы функции
            **kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения

        Raises:
            Exception: Если все варианты не удались
        """
        functions = [self.primary_func] + self.fallback_funcs

        for i, func in enumerate(functions):
            try:
                result = func(*args, **kwargs)

                # Обновление статистики
                if i == 0:
                    self.stats["primary_success"] += 1
                else:
                    self.stats["fallback_success"] += 1
                    self.stats["fallback_activations"] += 1
                    self.current_func_index = i

                return result

            except Exception as e:
                logger.warning(f"FallbackManager: Function {func.__name__} failed: {e}")

                if i == len(functions) - 1:
                    # Последняя функция не удалась
                    self.stats["total_failures"] += 1
                    raise

        # Не должно достигать этой точки
        raise Exception("All functions failed")

    async def execute_async(self, *args, **kwargs) -> T:
        """
        Асинхронное выполнение с запасными вариантами

        Args:
            *args: Аргументы функции
            **kwargs: Ключевые аргументы

        Returns:
            T: Результат выполнения
        """
        functions = [self.primary_func] + self.fallback_funcs

        for i, func in enumerate(functions):
            try:
                result = await func(*args, **kwargs)

                # Обновление статистики
                if i == 0:
                    self.stats["primary_success"] += 1
                else:
                    self.stats["fallback_success"] += 1
                    self.stats["fallback_activations"] += 1
                    self.current_func_index = i

                return result

            except Exception as e:
                logger.warning(f"FallbackManager: Async function {func.__name__} failed: {e}")

                if i == len(functions) - 1:
                    self.stats["total_failures"] += 1
                    raise

        raise Exception("All async functions failed")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        total_calls = (self.stats["primary_success"] +
                       self.stats["fallback_success"] +
                       self.stats["total_failures"])

        success_rate = ((self.stats["primary_success"] + self.stats["fallback_success"]) /
                        total_calls * 100 if total_calls > 0 else 0)

        return {
            **self.stats,
            "total_calls": total_calls,
            "success_rate_percent": success_rate,
            "current_function": self.current_func_index,
            "fallback_activation_rate": (
                self.stats["fallback_activations"] / total_calls * 100
                if total_calls > 0 else 0
            ),
            "timestamp": datetime.now().isoformat()
        }


class HealthMonitor:
    """Монитор здоровья для отказоустойчивости"""

    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.last_check: Dict[str, datetime] = {}

        # Пороги для определения состояния
        self.thresholds = {
            "response_time_ms": 1000,
            "error_rate_percent": 5.0,
            "check_interval_seconds": 60
        }

        logger.info("HealthMonitor: Инициализирован")

    def register_service(self, service_name: str,
                         health_check: Callable[[], bool],
                         dependencies: Optional[List[str]] = None):
        """
        Регистрация сервиса для мониторинга

        Args:
            service_name: Имя сервиса
            health_check: Функция проверки здоровья
            dependencies: Зависимые сервисы
        """
        self.services[service_name] = {
            "status": "unknown",
            "last_check": None,
            "response_time": None,
            "error_count": 0,
            "success_count": 0,
            "availability": 1.0,
            "dependencies": dependencies or []
        }

        self.health_checks[service_name] = health_check
        self.last_check[service_name] = datetime.now()

        logger.info(f"HealthMonitor: Зарегистрирован сервис '{service_name}'")

    def check_service(self, service_name: str) -> bool:
        """
        Проверка здоровья сервиса

        Args:
            service_name: Имя сервиса

        Returns:
            bool: True если сервис здоров
        """
        if service_name not in self.services:
            logger.warning(f"HealthMonitor: Неизвестный сервис '{service_name}'")
            return False

        try:
            start_time = time.time()

            # Проверка зависимостей
            dependencies = self.services[service_name]["dependencies"]
            for dep in dependencies:
                if not self.is_service_healthy(dep):
                    logger.warning(
                        f"HealthMonitor: Сервис '{service_name}' зависит от "
                        f"нездорового сервиса '{dep}'"
                    )
                    self._update_service_status(service_name, False, 0)
                    return False

            # Выполнение проверки здоровья
            health_check = self.health_checks[service_name]
            is_healthy = health_check()

            response_time = (time.time() - start_time) * 1000

            # Обновление статуса
            self._update_service_status(service_name, is_healthy, response_time)

            return is_healthy

        except Exception as e:
            logger.error(f"HealthMonitor: Ошибка проверки сервиса '{service_name}': {e}")
            self._update_service_status(service_name, False, 0)
            return False

    def check_all_services(self) -> Dict[str, bool]:
        """
        Проверка всех сервисов

        Returns:
            Dict: Статусы всех сервисов
        """
        results = {}

        for service_name in self.services:
            results[service_name] = self.check_service(service_name)

        return results

    def is_service_healthy(self, service_name: str) -> bool:
        """
        Проверка, здоров ли сервис (с учетом кэша)

        Args:
            service_name: Имя сервиса

        Returns:
            bool: True если сервис здоров
        """
        if service_name not in self.services:
            return False

        service_data = self.services[service_name]

        # Проверка времени последней проверки
        last_check = service_data["last_check"]
        if last_check:
            time_since_check = datetime.now() - last_check
            if time_since_check.total_seconds() > self.thresholds["check_interval_seconds"]:
                # Требуется новая проверка
                return self.check_service(service_name)

        return service_data["status"] == "healthy"

    def get_service_stats(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Получение статистики сервиса

        Args:
            service_name: Имя сервиса

        Returns:
            Optional[Dict]: Статистика сервиса
        """
        if service_name not in self.services:
            return None

        service_data = self.services[service_name]

        total_checks = service_data["error_count"] + service_data["success_count"]
        availability = service_data["availability"] * 100

        return {
            "service_name": service_name,
            "status": service_data["status"],
            "last_check": service_data["last_check"].isoformat() if service_data["last_check"] else None,
            "response_time_ms": service_data["response_time"],
            "availability_percent": availability,
            "total_checks": total_checks,
            "success_count": service_data["success_count"],
            "error_count": service_data["error_count"],
            "error_rate_percent": (
                service_data["error_count"] / total_checks * 100
                if total_checks > 0 else 0
            ),
            "dependencies": service_data["dependencies"],
            "timestamp": datetime.now().isoformat()
        }

    def get_overall_health(self) -> Dict[str, Any]:
        """
        Получение общего состояния здоровья системы

        Returns:
            Dict: Общее состояние
        """
        healthy_count = 0
        total_count = len(self.services)

        service_stats = {}

        for service_name in self.services:
            stats = self.get_service_stats(service_name)
            service_stats[service_name] = stats

            if stats and stats["status"] == "healthy":
                healthy_count += 1

        overall_health_percent = (healthy_count / total_count * 100) if total_count > 0 else 0

        return {
            "overall_status": "healthy" if overall_health_percent > 90 else "degraded",
            "overall_health_percent": overall_health_percent,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": service_stats,
            "timestamp": datetime.now().isoformat()
        }

    def _update_service_status(self, service_name: str, is_healthy: bool,
                               response_time: float):
        """Обновление статуса сервиса"""
        service_data = self.services[service_name]

        service_data["last_check"] = datetime.now()
        service_data["response_time"] = response_time

        if is_healthy:
            service_data["status"] = "healthy"
            service_data["success_count"] += 1

            # Проверка времени ответа
            if response_time > self.thresholds["response_time_ms"]:
                logger.warning(
                    f"HealthMonitor: Сервис '{service_name}' медленный: "
                    f"{response_time:.0f}ms"
                )
        else:
            service_data["status"] = "unhealthy"
            service_data["error_count"] += 1

        # Расчет доступности
        total = service_data["success_count"] + service_data["error_count"]
        if total > 0:
            service_data["availability"] = service_data["success_count"] / total

        # Проверка уровня ошибок
        error_rate = (service_data["error_count"] / total * 100) if total > 0 else 0
        if error_rate > self.thresholds["error_rate_percent"]:
            logger.error(
                f"HealthMonitor: Высокий уровень ошибок для сервиса '{service_name}': "
                f"{error_rate:.1f}%"
            )

    def start_periodic_checks(self, interval_seconds: int = 60):
        """
        Запуск периодических проверок

        Args:
            interval_seconds: Интервал проверок в секундах
        """
        import threading

        def check_loop():
            while True:
                try:
                    self.check_all_services()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"HealthMonitor: Ошибка в цикле проверок: {e}")
                    time.sleep(interval_seconds * 2)

        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()

        logger.info(f"HealthMonitor: Запущены периодические проверки каждые {interval_seconds} секунд")


class GracefulShutdown:
    """Менеджер корректного завершения"""

    def __init__(self):
        self.shutdown_requested = False
        self.shutdown_handlers: List[Callable] = []
        self.cleanup_handlers: List[Callable] = []

        # Регистрация обработчиков сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("GracefulShutdown: Инициализирован")

    def register_shutdown_handler(self, handler: Callable):
        """
        Регистрация обработчика завершения

        Args:
            handler: Функция-обработчик
        """
        self.shutdown_handlers.append(handler)
        logger.debug(f"GracefulShutdown: Зарегистрирован обработчик завершения: {handler.__name__}")

    def register_cleanup_handler(self, handler: Callable):
        """
        Регистрация обработчика очистки

        Args:
            handler: Функция-обработчик
        """
        self.cleanup_handlers.append(handler)
        logger.debug(f"GracefulShutdown: Зарегистрирован обработчик очистки: {handler.__name__}")

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        logger.warning(f"GracefulShutdown: Получен сигнал {signum}. Инициирую завершение...")
        self.shutdown_requested = True

        # Запуск завершения
        self.initiate_shutdown()

    def initiate_shutdown(self):
        """Инициирование корректного завершения"""
        logger.info("GracefulShutdown: Начинаю корректное завершение...")

        # Выполнение обработчиков завершения
        for handler in self.shutdown_handlers:
            try:
                logger.debug(f"GracefulShutdown: Выполняю обработчик завершения: {handler.__name__}")
                handler()
            except Exception as e:
                logger.error(f"GracefulShutdown: Ошибка в обработчике завершения: {e}")

        # Выполнение обработчиков очистки
        for handler in self.cleanup_handlers:
            try:
                logger.debug(f"GracefulShutdown: Выполняю обработчик очистки: {handler.__name__}")
                handler()
            except Exception as e:
                logger.error(f"GracefulShutdown: Ошибка в обработчике очистки: {e}")

        logger.info("GracefulShutdown: Корректное завершение выполнено")

        # Выход из приложения
        sys.exit(0)

    def wait_for_shutdown(self):
        """Ожидание запроса на завершение"""
        while not self.shutdown_requested:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.shutdown_requested = True
                self.initiate_shutdown()

    def is_shutdown_requested(self) -> bool:
        """Проверка, запрошено ли завершение"""
        return self.shutdown_requested