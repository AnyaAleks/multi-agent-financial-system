"""
Мониторинг здоровья системы
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import socket
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Prometheus метрики
REQUEST_COUNT = Counter(
    'financial_system_requests_total',
    'Total number of requests',
    ['agent', 'status']
)

REQUEST_LATENCY = Histogram(
    'financial_system_request_latency_seconds',
    'Request latency in seconds',
    ['agent']
)

AGENT_HEALTH = Gauge(
    'financial_system_agent_health',
    'Agent health status (1 = healthy, 0 = unhealthy)',
    ['agent']
)

SYSTEM_RESOURCES = Gauge(
    'financial_system_resources',
    'System resource usage',
    ['resource_type']
)

ERROR_COUNT = Counter(
    'financial_system_errors_total',
    'Total number of errors',
    ['agent', 'error_type']
)


class HealthMonitor:
    """Монитор здоровья системы"""

    def __init__(self):
        self.metrics_port = settings.metrics_port
        self.agents_health: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000,
            "error_rate": 0.05,  # 5%
            "agent_timeout": 30  # секунд
        }

        self.alerts: List[Dict[str, Any]] = []
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {
            "system": [],
            "agents": {},
            "performance": []
        }

        # Инициализация мониторинга агентов
        self._initialize_agents_monitoring()

    def _initialize_agents_monitoring(self):
        """Инициализация мониторинга агентов"""
        self.agents_health = {
            "data_agent": {
                "status": "unknown",
                "last_check": None,
                "response_time": None,
                "error_count": 0,
                "success_count": 0,
                "availability": 1.0,
                "custom_metrics": {}
            },
            "analysis_agent": {
                "status": "unknown",
                "last_check": None,
                "response_time": None,
                "error_count": 0,
                "success_count": 0,
                "availability": 1.0,
                "custom_metrics": {}
            },
            "report_agent": {
                "status": "unknown",
                "last_check": None,
                "response_time": None,
                "error_count": 0,
                "success_count": 0,
                "availability": 1.0,
                "custom_metrics": {}
            },
            "manager_agent": {
                "status": "unknown",
                "last_check": None,
                "response_time": None,
                "error_count": 0,
                "success_count": 0,
                "availability": 1.0,
                "custom_metrics": {}
            }
        }

        for agent in self.agents_health:
            self.metrics_history["agents"][agent] = []

    def start(self):
        """Запуск мониторинга"""
        logger.info(f"HealthMonitor: Запуск мониторинга на порту {self.metrics_port}")

        # Запуск Prometheus сервера
        start_http_server(self.metrics_port)

        # Запуск фоновых задач
        asyncio.create_task(self._monitor_system_resources())
        asyncio.create_task(self._monitor_agents_health())
        asyncio.create_task(self._cleanup_old_metrics())

        logger.info("HealthMonitor: Мониторинг запущен")

    def stop(self):
        """Остановка мониторинга"""
        logger.info("HealthMonitor: Остановка мониторинга")

    async def _monitor_system_resources(self):
        """Мониторинг системных ресурсов"""
        while True:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics = metrics

                # Сохранение в историю
                self.metrics_history["system"].append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                })

                # Проверка порогов
                self._check_system_thresholds(metrics)

                # Обновление Prometheus метрик
                self._update_prometheus_metrics(metrics)

                await asyncio.sleep(30)  # Каждые 30 секунд

            except Exception as e:
                logger.error(f"HealthMonitor: Ошибка мониторинга ресурсов: {e}")
                await asyncio.sleep(60)

    async def _monitor_agents_health(self):
        """Мониторинг здоровья агентов"""
        while True:
            try:
                for agent_name in self.agents_health.keys():
                    await self._check_agent_health(agent_name)

                await asyncio.sleep(60)  # Каждую минуту

            except Exception as e:
                logger.error(f"HealthMonitor: Ошибка мониторинга агентов: {e}")
                await asyncio.sleep(120)

    async def _check_agent_health(self, agent_name: str):
        """Проверка здоровья конкретного агента"""
        try:
            start_time = time.time()

            # Здесь должна быть реальная проверка здоровья агента
            # Для примера используем имитацию

            if agent_name == "data_agent":
                # Проверка доступности MCP сервера
                health_status = await self._check_mcp_server_health()
            elif agent_name == "analysis_agent":
                # Проверка доступности LLM
                health_status = await self._check_llm_health()
            else:
                # Базовая проверка
                health_status = {"status": "healthy", "response_time": 0.1}

            response_time = (time.time() - start_time) * 1000  # в мс

            # Обновление состояния агента
            agent_data = self.agents_health[agent_name]
            agent_data["last_check"] = datetime.now().isoformat()
            agent_data["response_time"] = response_time

            if health_status["status"] == "healthy":
                agent_data["status"] = "healthy"
                agent_data["success_count"] += 1
                AGENT_HEALTH.labels(agent=agent_name).set(1)
            else:
                agent_data["status"] = "unhealthy"
                agent_data["error_count"] += 1
                AGENT_HEALTH.labels(agent=agent_name).set(0)
                self._create_alert(
                    f"Agent {agent_name} is unhealthy",
                    f"Status: {health_status.get('details', 'Unknown error')}",
                    "agent_health"
                )

            # Расчет доступности
            total = agent_data["success_count"] + agent_data["error_count"]
            if total > 0:
                agent_data["availability"] = agent_data["success_count"] / total

            # Сохранение в историю
            self.metrics_history["agents"][agent_name].append({
                "timestamp": datetime.now().isoformat(),
                "status": agent_data["status"],
                "response_time": response_time,
                "availability": agent_data["availability"]
            })

            # Проверка порогов
            if response_time > self.alert_thresholds["response_time_ms"]:
                self._create_alert(
                    f"High response time for {agent_name}",
                    f"Response time: {response_time:.0f}ms (threshold: {self.alert_thresholds['response_time_ms']}ms)",
                    "performance"
                )

            logger.debug(f"HealthMonitor: Проверка {agent_name}: {agent_data['status']}")

        except Exception as e:
            logger.error(f"HealthMonitor: Ошибка проверки агента {agent_name}: {e}")
            self.agents_health[agent_name]["status"] = "error"
            AGENT_HEALTH.labels(agent=agent_name).set(0)

    async def _check_mcp_server_health(self) -> Dict[str, Any]:
        """Проверка здоровья MCP сервера"""
        try:
            # Попытка подключения к MCP серверу
            host = settings.mcp_financial_host
            port = settings.mcp_financial_port

            # Создание сокета
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                return {"status": "healthy", "details": "MCP server is reachable"}
            else:
                return {"status": "unhealthy", "details": f"MCP server unreachable: {result}"}

        except Exception as e:
            return {"status": "unhealthy", "details": f"MCP check error: {str(e)}"}

    async def _check_llm_health(self) -> Dict[str, Any]:
        """Проверка здоровья LLM сервиса"""
        try:
            # Простая проверка доступности LLM
            # В реальной системе здесь должен быть вызов API
            return {"status": "healthy", "details": "LLM service is available"}
        except Exception as e:
            return {"status": "unhealthy", "details": f"LLM check error: {str(e)}"}

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Сбор системных метрик"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Память
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Диск
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            # Сеть
            net_io = psutil.net_io_counters()

            # Процессы
            processes = len(psutil.pids())

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "percent_per_core": psutil.cpu_percent(interval=0, percpu=True)
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                    "read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                    "write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
                },
                "network": {
                    "bytes_sent_mb": net_io.bytes_sent / (1024**2),
                    "bytes_recv_mb": net_io.bytes_recv / (1024**2),
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "system": {
                    "processes": processes,
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    "uptime_hours": (time.time() - psutil.boot_time()) / 3600
                }
            }

            return metrics

        except Exception as e:
            logger.error(f"HealthMonitor: Ошибка сбора системных метрик: {e}")
            return {}

    def _check_system_thresholds(self, metrics: Dict[str, Any]):
        """Проверка системных метрик на превышение порогов"""
        alerts = []

        # Проверка CPU
        cpu_percent = metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu_high",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "value": cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"]
            })

        # Проверка памяти
        memory_percent = metrics.get("memory", {}).get("percent", 0)
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "memory_high",
                "message": f"High memory usage: {memory_percent:.1f}%",
                "value": memory_percent,
                "threshold": self.alert_thresholds["memory_percent"]
            })

        # Проверка диска
        disk_percent = metrics.get("disk", {}).get("percent", 0)
        if disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "disk_high",
                "message": f"High disk usage: {disk_percent:.1f}%",
                "value": disk_percent,
                "threshold": self.alert_thresholds["disk_percent"]
            })

        # Создание алертов
        for alert in alerts:
            self._create_alert(
                alert["message"],
                f"Current: {alert['value']:.1f}%, Threshold: {alert['threshold']}%",
                alert["type"]
            )

    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Обновление Prometheus метрик"""
        try:
            # CPU
            SYSTEM_RESOURCES.labels(resource_type='cpu_percent').set(
                metrics.get("cpu", {}).get("percent", 0)
            )

            # Память
            SYSTEM_RESOURCES.labels(resource_type='memory_percent').set(
                metrics.get("memory", {}).get("percent", 0)
            )

            # Диск
            SYSTEM_RESOURCES.labels(resource_type='disk_percent').set(
                metrics.get("disk", {}).get("percent", 0)
            )

            # Процессы
            SYSTEM_RESOURCES.labels(resource_type='processes').set(
                metrics.get("system", {}).get("processes", 0)
            )

        except Exception as e:
            logger.error(f"HealthMonitor: Ошибка обновления Prometheus метрик: {e}")

    def _create_alert(self, title: str, description: str, alert_type: str):
        """Создание алерта"""
        alert = {
            "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": title,
            "description": description,
            "type": alert_type,
            "severity": self._get_alert_severity(alert_type),
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
            "resolved": False
        }

        self.alerts.append(alert)
        logger.warning(f"HealthMonitor: Новый алерт: {title} - {description}")

        # Ограничение истории алертов
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def _get_alert_severity(self, alert_type: str) -> str:
        """Определение серьезности алерта"""
        if alert_type in ["cpu_high", "memory_high", "disk_high"]:
            return "warning"
        elif alert_type == "agent_health":
            return "critical"
        else:
            return "info"

    async def _cleanup_old_metrics(self):
        """Очистка старых метрик"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                cutoff_str = cutoff_time.isoformat()

                # Очистка системных метрик
                self.metrics_history["system"] = [
                    m for m in self.metrics_history["system"]
                    if m["timestamp"] > cutoff_str
                ]

                # Очистка метрик агентов
                for agent in self.metrics_history["agents"]:
                    self.metrics_history["agents"][agent] = [
                        m for m in self.metrics_history["agents"][agent]
                        if m["timestamp"] > cutoff_str
                    ]

                # Очистка старых алертов
                self.alerts = [
                    a for a in self.alerts
                    if datetime.fromisoformat(a["timestamp"]) > cutoff_time
                ]

                await asyncio.sleep(3600)  # Каждый час

            except Exception as e:
                logger.error(f"HealthMonitor: Ошибка очистки метрик: {e}")
                await asyncio.sleep(7200)

    def check_agent_health(self, agent_name: str) -> bool:
        """
        Проверка здоровья агента

        Args:
            agent_name: Имя агента

        Returns:
            bool: True если агент здоров
        """
        agent_data = self.agents_health.get(agent_name)
        if not agent_data:
            return False

        # Агент считается здоровым, если:
        # 1. Статус "healthy"
        # 2. Последняя проверка была не более 2 минут назад
        # 3. Доступность выше 95%

        if agent_data["status"] != "healthy":
            return False

        if agent_data["last_check"]:
            last_check = datetime.fromisoformat(agent_data["last_check"])
            if (datetime.now() - last_check).total_seconds() > 120:
                return False

        if agent_data.get("availability", 0) < 0.95:
            return False

        return True

    def record_request(self, agent: str, status: str, latency: float):
        """
        Запись информации о запросе

        Args:
            agent: Имя агента
            status: Статус запроса
            latency: Время выполнения в секундах
        """
        REQUEST_COUNT.labels(agent=agent, status=status).inc()
        REQUEST_LATENCY.labels(agent=agent).observe(latency)

    def record_error(self, agent: str, error_type: str):
        """
        Запись информации об ошибке

        Args:
            agent: Имя агента
            error_type: Тип ошибки
        """
        ERROR_COUNT.labels(agent=agent, error_type=error_type).inc()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Получение общего статуса здоровья системы

        Returns:
            Dict: Статус здоровья системы
        """
        healthy_agents = sum(1 for agent in self.agents_health.values()
                           if agent["status"] == "healthy")
        total_agents = len(self.agents_health)

        # Расчет общего статуса системы
        system_status = "healthy"
        if healthy_agents < total_agents * 0.5:  # Менее 50% агентов здоровы
            system_status = "critical"
        elif healthy_agents < total_agents * 0.8:  # Менее 80% агентов здоровы
            system_status = "degraded"

        # Активные алерты
        active_alerts = [a for a in self.alerts if not a["resolved"]]

        return {
            "system_status": system_status,
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "total": total_agents,
                "healthy": healthy_agents,
                "unhealthy": total_agents - healthy_agents,
                "details": self.agents_health
            },
            "alerts": {
                "total": len(active_alerts),
                "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
                "warning": len([a for a in active_alerts if a["severity"] == "warning"]),
                "list": active_alerts[-10:]  # Последние 10 алертов
            },
            "system_metrics": self.system_metrics,
            "performance": {
                "request_count": self._get_prometheus_metric_value(REQUEST_COUNT),
                "error_rate": self._calculate_error_rate(),
                "average_latency": self._calculate_average_latency()
            }
        }

    def _get_prometheus_metric_value(self, metric) -> float:
        """Получение значения Prometheus метрики"""
        try:
            # В реальной системе здесь должен быть запрос к Prometheus API
            return 0.0
        except:
            return 0.0

    def _calculate_error_rate(self) -> float:
        """Расчет уровня ошибок"""
        try:
            total_requests = sum(self._get_prometheus_metric_value(REQUEST_COUNT.labels(agent=a, status=s))
                               for a in self.agents_health.keys()
                               for s in ["success", "error"])

            total_errors = sum(self._get_prometheus_metric_value(REQUEST_COUNT.labels(agent=a, status="error"))
                             for a in self.agents_health.keys())

            if total_requests == 0:
                return 0.0

            return total_errors / total_requests

        except:
            return 0.0

    def _calculate_average_latency(self) -> float:
        """Расчет средней задержки"""
        try:
            # В реальной системе здесь должен быть расчет на основе метрик
            return 0.0
        except:
            return 0.0

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Подтверждение алерта

        Args:
            alert_id: ID алерта

        Returns:
            bool: Успешно ли подтверждено
        """
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                logger.info(f"HealthMonitor: Алерт {alert_id} подтвержден")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Разрешение алерта

        Args:
            alert_id: ID алерта

        Returns:
            bool: Успешно ли разрешено
        """
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now().isoformat()
                logger.info(f"HealthMonitor: Алерт {alert_id} разрешен")
                return True
        return False

    def get_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """
        Получение истории метрик

        Args:
            hours: Количество часов истории

        Returns:
            Dict: История метрик
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        filtered_history = {
            "system": [
                m for m in self.metrics_history["system"]
                if m["timestamp"] > cutoff_str
            ],
            "agents": {},
            "alerts": [
                a for a in self.alerts
                if datetime.fromisoformat(a["timestamp"]) > cutoff_time
            ]
        }

        for agent, history in self.metrics_history["agents"].items():
            filtered_history["agents"][agent] = [
                m for m in history
                if m["timestamp"] > cutoff_str
            ]

        return filtered_history

    def export_metrics(self) -> Dict[str, Any]:
        """
        Экспорт метрик для внешних систем

        Returns:
            Dict: Метрики в формате для экспорта
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": self.get_health_status(),
            "system_metrics": self.system_metrics,
            "agents_health": self.agents_health,
            "alerts_summary": {
                "total": len(self.alerts),
                "active": len([a for a in self.alerts if not a["resolved"]]),
                "acknowledged": len([a for a in self.alerts if a["acknowledged"]]),
                "resolved": len([a for a in self.alerts if a["resolved"]])
            }
        }