"""
Параллельная обработка и оптимизация производительности
"""
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import time
from functools import wraps
import threading
from queue import Queue
import numpy as np

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """Менеджер параллельной обработки"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or settings.max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="financial_processor"
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(2, self.max_workers // 2)
        )

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.performance_stats: Dict[str, List[float]] = {
            "execution_times": [],
            "throughput": [],
            "error_rates": []
        }

        # Кэш для результатов
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()

        # Запуск фоновых обработчиков
        self._start_background_workers()

    def _start_background_workers(self):
        """Запуск фоновых обработчиков"""
        self.worker_threads = []

        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"worker_{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)

        logger.info(f"ParallelProcessor: Запущено {self.max_workers} воркеров")

    def _worker_loop(self):
        """Цикл обработки задач воркером"""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # Сигнал завершения
                    break

                task_id, func, args, kwargs = task

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Сохранение результата
                    self.result_queue.put({
                        "task_id": task_id,
                        "result": result,
                        "status": "success",
                        "execution_time": execution_time,
                        "worker": threading.current_thread().name
                    })

                    # Обновление статистики
                    with self.cache_lock:
                        self.performance_stats["execution_times"].append(execution_time)
                        if len(self.performance_stats["execution_times"]) > 1000:
                            self.performance_stats["execution_times"] = \
                                self.performance_stats["execution_times"][-1000:]

                except Exception as e:
                    self.result_queue.put({
                        "task_id": task_id,
                        "result": None,
                        "status": "error",
                        "error": str(e),
                        "execution_time": 0,
                        "worker": threading.current_thread().name
                    })

                finally:
                    self.task_queue.task_done()

            except Exception as e:
                logger.error(f"ParallelProcessor: Ошибка воркера: {e}")

    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """
        Отправка задачи на выполнение

        Args:
            func: Функция для выполнения
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            str: ID задачи
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(args) + str(kwargs))}"

        # Проверка кэша
        cache_key = self._generate_cache_key(func, args, kwargs)
        with self.cache_lock:
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result["timestamp"] < 300:  # 5 минут TTL
                    logger.debug(f"ParallelProcessor: Использую кэшированный результат для {task_id}")

                    # Возвращаем кэшированный результат синхронно
                    self.result_queue.put({
                        "task_id": task_id,
                        "result": cached_result["result"],
                        "status": "cached",
                        "execution_time": 0.001,
                        "worker": "cache"
                    })

                    return task_id

        # Добавление задачи в очередь
        self.task_queue.put((task_id, func, args, kwargs))

        # Сохранение информации о задаче
        self.active_tasks[task_id] = {
            "submitted_at": datetime.now().isoformat(),
            "function": func.__name__,
            "status": "pending",
            "cache_key": cache_key
        }

        logger.debug(f"ParallelProcessor: Задача {task_id} отправлена в очередь")
        return task_id

    def submit_batch(self, func: Callable, items: List[Any],
                     batch_size: int = 10) -> List[str]:
        """
        Отправка пакета задач

        Args:
            func: Функция для выполнения
            items: Список элементов для обработки
            batch_size: Размер пакета

        Returns:
            List[str]: Список ID задач
        """
        task_ids = []

        # Разделение на пакеты
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Создание задачи для пакета
            task_id = self.submit_task(
                self._process_batch,
                func, batch
            )
            task_ids.append(task_id)

        logger.info(f"ParallelProcessor: Отправлен пакет из {len(items)} элементов "
                    f"в {len(task_ids)} задачах")

        return task_ids

    def wait_for_results(self, task_ids: List[str],
                         timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Ожидание результатов задач

        Args:
            task_ids: Список ID задач
            timeout: Таймаут в секундах

        Returns:
            Dict: Результаты задач
        """
        results = {}
        start_time = time.time()

        while task_ids and (timeout is None or time.time() - start_time < timeout):
            try:
                # Проверка результатов в очереди
                if not self.result_queue.empty():
                    result = self.result_queue.get_nowait()

                    task_id = result["task_id"]
                    if task_id in task_ids:
                        results[task_id] = result

                        # Обновление статуса задачи
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id].update({
                                "completed_at": datetime.now().isoformat(),
                                "status": result["status"],
                                "execution_time": result.get("execution_time", 0)
                            })

                            # Кэширование успешных результатов
                            if result["status"] == "success":
                                cache_key = self.active_tasks[task_id].get("cache_key")
                                if cache_key:
                                    with self.cache_lock:
                                        self.cache[cache_key] = {
                                            "result": result["result"],
                                            "timestamp": time.time()
                                        }

                        task_ids.remove(task_id)
                        self.result_queue.task_done()

                else:
                    # Небольшая пауза для уменьшения нагрузки на CPU
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"ParallelProcessor: Ошибка ожидания результатов: {e}")
                break

        # Проверка таймаута
        if task_ids and timeout and time.time() - start_time >= timeout:
            logger.warning(f"ParallelProcessor: Таймаут ожидания для {len(task_ids)} задач")

            for task_id in task_ids:
                results[task_id] = {
                    "task_id": task_id,
                    "result": None,
                    "status": "timeout",
                    "error": "Execution timeout",
                    "execution_time": 0
                }

        return results

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Асинхронное выполнение функции

        Args:
            func: Функция для выполнения
            args: Аргументы функции
            kwargs: Ключевые аргументы

        Returns:
            Any: Результат выполнения
        """
        loop = asyncio.get_event_loop()

        try:
            # Проверка кэша
            cache_key = self._generate_cache_key(func, args, kwargs)
            with self.cache_lock:
                if cache_key in self.cache:
                    cached = self.cache[cache_key]
                    if time.time() - cached["timestamp"] < 300:
                        logger.debug("ParallelProcessor: Использую кэшированный результат (async)")
                        return cached["result"]

            # Асинхронное выполнение в thread pool
            result = await loop.run_in_executor(
                self.executor,
                lambda: func(*args, **kwargs)
            )

            # Кэширование результата
            with self.cache_lock:
                self.cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

            return result

        except Exception as e:
            logger.error(f"ParallelProcessor: Ошибка асинхронного выполнения: {e}")
            raise

    def execute_parallel(self, tasks: List[Dict[str, Any]],
                         strategy: str = "fan_out") -> Dict[str, Any]:
        """
        Параллельное выполнение задач

        Args:
            tasks: Список задач
            strategy: Стратегия выполнения (fan_out, map_reduce, pipeline)

        Returns:
            Dict: Результаты выполнения
        """
        start_time = time.time()

        if strategy == "fan_out":
            results = self._execute_fan_out(tasks)
        elif strategy == "map_reduce":
            results = self._execute_map_reduce(tasks)
        elif strategy == "pipeline":
            results = self._execute_pipeline(tasks)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        execution_time = time.time() - start_time

        return {
            "strategy": strategy,
            "results": results,
            "execution_time": execution_time,
            "task_count": len(tasks),
            "throughput": len(tasks) / execution_time if execution_time > 0 else 0
        }

    def _execute_fan_out(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Стратегия Fan-Out (параллельное выполнение всех задач)"""
        task_ids = []

        # Отправка всех задач
        for task in tasks:
            task_id = self.submit_task(
                task["func"],
                *task.get("args", []),
                **task.get("kwargs", {})
            )
            task_ids.append(task_id)

        # Ожидание результатов
        results = self.wait_for_results(task_ids, timeout=60)

        # Сбор результатов в правильном порядке
        ordered_results = []
        for task_id in task_ids:
            if task_id in results:
                ordered_results.append(results[task_id]["result"])
            else:
                ordered_results.append(None)

        return ordered_results

    def _execute_map_reduce(self, tasks: List[Dict[str, Any]]) -> Any:
        """Стратегия Map-Reduce"""
        if len(tasks) != 2:
            raise ValueError("Map-Reduce requires exactly 2 tasks: map and reduce")

        map_task = tasks[0]
        reduce_task = tasks[1]

        # Выполнение map фазы
        map_results = self._execute_fan_out([map_task])

        # Выполнение reduce фазы
        reduce_result = reduce_task["func"](map_results, *reduce_task.get("args", []))

        return reduce_result

    def _execute_pipeline(self, tasks: List[Dict[str, Any]]) -> Any:
        """Стратегия Pipeline (последовательная обработка)"""
        current_result = None

        for task in tasks:
            if current_result is None:
                # Первая задача
                task_id = self.submit_task(
                    task["func"],
                    *task.get("args", []),
                    **task.get("kwargs", {})
                )
                result = self.wait_for_results([task_id], timeout=30)
                if task_id in result:
                    current_result = result[task_id]["result"]
                else:
                    return None
            else:
                # Последующие задачи получают результат предыдущей
                task_id = self.submit_task(
                    task["func"],
                    current_result,
                    *task.get("args", []),
                    **task.get("kwargs", {})
                )
                result = self.wait_for_results([task_id], timeout=30)
                if task_id in result:
                    current_result = result[task_id]["result"]
                else:
                    return None

        return current_result

    def _process_batch(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Обработка пакета элементов

        Args:
            func: Функция обработки
            items: Список элементов

        Returns:
            List: Результаты обработки
        """
        results = []

        for item in items:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"ParallelProcessor: Ошибка обработки элемента: {e}")
                results.append(None)

        return results

    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Генерация ключа кэша"""
        import hashlib
        import pickle

        try:
            # Сериализация функции и аргументов
            key_data = {
                "func_name": func.__name__,
                "args": args,
                "kwargs": kwargs
            }

            serialized = pickle.dumps(key_data)
            return hashlib.sha256(serialized).hexdigest()

        except:
            # Fallback для случаев, когда сериализация невозможна
            return f"{func.__name__}_{hash(str(args) + str(kwargs))}"

    def clear_cache(self):
        """Очистка кэша"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("ParallelProcessor: Кэш очищен")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности

        Returns:
            Dict: Статистика производительности
        """
        with self.cache_lock:
            exec_times = self.performance_stats["execution_times"]

            if exec_times:
                avg_time = np.mean(exec_times)
                p95_time = np.percentile(exec_times, 95)
                p99_time = np.percentile(exec_times, 99)
            else:
                avg_time = p95_time = p99_time = 0

        return {
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "cache_size": len(self.cache),
            "thread_pool_workers": self.max_workers,
            "performance": {
                "average_execution_time": avg_time,
                "p95_execution_time": p95_time,
                "p99_execution_time": p99_time,
                "total_tasks_processed": len(exec_times)
            },
            "timestamp": datetime.now().isoformat()
        }

    def shutdown(self):
        """Корректное завершение работы"""
        logger.info("ParallelProcessor: Завершение работы...")

        # Остановка воркеров
        for _ in range(self.max_workers):
            self.task_queue.put(None)

        # Ожидание завершения воркеров
        for thread in self.worker_threads:
            thread.join(timeout=5)

        # Завершение исполнителей
        self.executor.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        logger.info("ParallelProcessor: Работа завершена")


def parallelize(func=None, *, max_workers=None, cache_results=True, timeout=30):
    """
    Декоратор для параллельного выполнения функций

    Args:
        max_workers: Максимальное количество воркеров
        cache_results: Кэшировать ли результаты
        timeout: Таймаут выполнения

    Returns:
        Декорированная функция
    """

    def decorator(original_func):
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            # Создание процессора при первом вызове
            if not hasattr(wrapper, '_processor'):
                wrapper._processor = ParallelProcessor(max_workers=max_workers)

            processor = wrapper._processor

            try:
                # Параллельное выполнение
                result = asyncio.run(
                    processor.execute_async(original_func, *args, **kwargs)
                )
                return result

            except Exception as e:
                logger.error(f"Parallelize decorator error for {original_func.__name__}: {e}")

                # Fallback: последовательное выполнение
                logger.warning(f"Using fallback sequential execution for {original_func.__name__}")
                return original_func(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


class DataParallelCollector:
    """Параллельный сборщик данных"""

    def __init__(self, processor: ParallelProcessor):
        self.processor = processor

    async def collect_parallel(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Параллельный сбор данных из нескольких источников

        Args:
            sources: Список источников данных

        Returns:
            Dict: Собранные данные
        """
        tasks = []

        for source in sources:
            tasks.append({
                "func": self._fetch_from_source,
                "args": (source,),
                "kwargs": {}
            })

        # Параллельное выполнение
        results = self.processor.execute_parallel(tasks, strategy="fan_out")

        # Агрегация результатов
        aggregated = {}
        for i, result in enumerate(results["results"]):
            if result:
                source_name = sources[i].get("name", f"source_{i}")
                aggregated[source_name] = result

        return {
            "sources_queried": len(sources),
            "sources_successful": len(aggregated),
            "data": aggregated,
            "collection_time": results["execution_time"]
        }

    def _fetch_from_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Получение данных из источника"""
        try:
            source_type = source.get("type")

            if source_type == "yahoo_finance":
                return self._fetch_yahoo_finance(source)
            elif source_type == "news_api":
                return self._fetch_news_api(source)
            elif source_type == "sec_edgar":
                return self._fetch_sec_edgar(source)
            else:
                logger.warning(f"Unknown source type: {source_type}")
                return None

        except Exception as e:
            logger.error(f"Error fetching from source {source.get('name')}: {e}")
            return None

    def _fetch_yahoo_finance(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Получение данных из Yahoo Finance"""
        # Имитация получения данных
        import yfinance as yf

        symbol = source.get("symbol", "AAPL")
        period = source.get("period", "1mo")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        return {
            "source": "yahoo_finance",
            "symbol": symbol,
            "data_type": "price_history",
            "data_points": len(hist),
            "period": period,
            "timestamp": datetime.now().isoformat()
        }

    def _fetch_news_api(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Получение новостей из NewsAPI"""
        # Имитация получения данных
        query = source.get("query", "finance")
        days = source.get("days", 7)

        return {
            "source": "news_api",
            "query": query,
            "data_type": "news_articles",
            "articles_count": 10,  # Имитация
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }

    def _fetch_sec_edgar(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Получение данных из SEC EDGAR"""
        # Имитация получения данных
        company = source.get("company", "Apple Inc.")
        filing_type = source.get("filing_type", "10-K")

        return {
            "source": "sec_edgar",
            "company": company,
            "filing_type": filing_type,
            "data_type": "financial_statements",
            "timestamp": datetime.now().isoformat()
        }