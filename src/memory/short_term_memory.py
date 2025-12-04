"""
Кратковременная память системы
"""
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
import hashlib

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShortTermMemory:
    """Кратковременная память на основе Redis"""

    def __init__(self):
        self.redis_client = None
        self.ttl = settings.redis_ttl
        self._connect_redis()

    def _connect_redis(self):
        """Подключение к Redis"""
        try:
            self.redis_client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=False,  # Для бинарных данных
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Тест подключения
            self.redis_client.ping()
            logger.info(f"ShortTermMemory: Подключено к Redis: {settings.redis_url}")

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка подключения к Redis: {e}")
            self.redis_client = None

    def store(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Сохранение данных в память

        Args:
            key: Ключ для хранения
            data: Данные для сохранения
            ttl: Время жизни в секундах (по умолчанию из настроек)

        Returns:
            bool: Успешно ли сохранено
        """
        if not self.redis_client:
            logger.warning("ShortTermMemory: Redis не доступен, данные не сохранены")
            return False

        try:
            # Сериализация данных
            serialized_data = pickle.dumps({
                "data": data,
                "metadata": {
                    "stored_at": datetime.now().isoformat(),
                    "ttl": ttl or self.ttl,
                    "data_type": type(data).__name__,
                    "size_bytes": len(pickle.dumps(data))
                }
            })

            # Сохранение в Redis
            expiration = ttl or self.ttl
            success = self.redis_client.setex(
                f"stm:{key}",
                expiration,
                serialized_data
            )

            if success:
                logger.debug(f"ShortTermMemory: Данные сохранены по ключу {key}")
                return True
            else:
                logger.warning(f"ShortTermMemory: Не удалось сохранить данные по ключу {key}")
                return False

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка сохранения данных: {e}")
            return False

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Получение данных из памяти

        Args:
            key: Ключ данных

        Returns:
            Optional[Dict]: Данные или None если не найдены
        """
        if not self.redis_client:
            logger.warning("ShortTermMemory: Redis не доступен, данные не получены")
            return None

        try:
            # Получение данных из Redis
            serialized_data = self.redis_client.get(f"stm:{key}")

            if not serialized_data:
                logger.debug(f"ShortTermMemory: Данные не найдены по ключу {key}")
                return None

            # Десериализация
            stored_data = pickle.loads(serialized_data)
            data = stored_data["data"]

            # Обновление метаданных о доступе
            metadata = stored_data.get("metadata", {})
            metadata["last_accessed"] = datetime.now().isoformat()
            metadata["access_count"] = metadata.get("access_count", 0) + 1

            # Обновление в Redis
            self._update_metadata(key, metadata)

            logger.debug(f"ShortTermMemory: Данные получены по ключу {key}")
            return data

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка получения данных: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Удаление данных из памяти"""
        if not self.redis_client:
            return False

        try:
            deleted = self.redis_client.delete(f"stm:{key}")
            if deleted:
                logger.debug(f"ShortTermMemory: Данные удалены по ключу {key}")
            return deleted > 0
        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка удаления данных: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Проверка существования данных"""
        if not self.redis_client:
            return False

        try:
            return self.redis_client.exists(f"stm:{key}") > 0
        except:
            return False

    def search(self, pattern: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Поиск данных по шаблону

        Args:
            pattern: Шаблон для поиска (например, "task_*")
            limit: Максимальное количество результатов

        Returns:
            List[Dict]: Список найденных данных
        """
        if not self.redis_client:
            return []

        try:
            keys = self.redis_client.keys(f"stm:{pattern}")
            results = []

            for key in keys[:limit]:
                data = self.retrieve(key.decode().replace("stm:", ""))
                if data:
                    results.append({
                        "key": key.decode().replace("stm:", ""),
                        "data": data
                    })

            return results

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка поиска: {e}")
            return []

    def store_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Сохранение сессионных данных

        Args:
            session_id: ID сессии
            data: Данные сессии

        Returns:
            bool: Успешно ли сохранено
        """
        session_key = f"session:{session_id}"
        session_data = {
            "session_id": session_id,
            "data": data,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }

        return self.store(session_key, session_data, ttl=3600)  # 1 час TTL

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение сессионных данных

        Args:
            session_id: ID сессии

        Returns:
            Optional[Dict]: Данные сессии
        """
        session_key = f"session:{session_id}"
        session_data = self.retrieve(session_key)

        if session_data:
            # Обновление времени последней активности
            session_data["last_activity"] = datetime.now().isoformat()
            self.store(session_key, session_data, ttl=3600)

            return session_data.get("data")

        return None

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Обновление сессионных данных

        Args:
            session_id: ID сессии
            updates: Обновления для данных сессии

        Returns:
            bool: Успешно ли обновлено
        """
        current_data = self.get_session(session_id) or {}
        current_data.update(updates)

        return self.store_session(session_id, current_data)

    def store_conversation_context(self, conversation_id: str,
                                   messages: List[Dict[str, Any]]) -> bool:
        """
        Сохранение контекста разговора

        Args:
            conversation_id: ID разговора
            messages: Список сообщений

        Returns:
            bool: Успешно ли сохранено
        """
        context_key = f"conversation:{conversation_id}"
        context_data = {
            "conversation_id": conversation_id,
            "messages": messages[-10:],  # Сохраняем последние 10 сообщений
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages)
        }

        return self.store(context_key, context_data, ttl=1800)  # 30 минут TTL

    def get_conversation_context(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получение контекста разговора

        Args:
            conversation_id: ID разговора

        Returns:
            Optional[List]: Список сообщений
        """
        context_key = f"conversation:{conversation_id}"
        context_data = self.retrieve(context_key)

        if context_data:
            return context_data.get("messages", [])

        return None

    def add_to_conversation(self, conversation_id: str,
                            message: Dict[str, Any]) -> bool:
        """
        Добавление сообщения в разговор

        Args:
            conversation_id: ID разговора
            message: Сообщение для добавления

        Returns:
            bool: Успешно ли добавлено
        """
        current_messages = self.get_conversation_context(conversation_id) or []
        current_messages.append(message)

        return self.store_conversation_context(conversation_id, current_messages)

    def store_analysis_context(self, analysis_id: str,
                               context: Dict[str, Any]) -> bool:
        """
        Сохранение контекста анализа

        Args:
            analysis_id: ID анализа
            context: Контекст анализа

        Returns:
            bool: Успешно ли сохранено
        """
        analysis_key = f"analysis:{analysis_id}"
        analysis_data = {
            "analysis_id": analysis_id,
            "context": context,
            "created_at": datetime.now().isoformat(),
            "data_hash": self._calculate_data_hash(context)
        }

        return self.store(analysis_key, analysis_data, ttl=7200)  # 2 часа TTL

    def get_analysis_context(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение контекста анализа

        Args:
            analysis_id: ID анализа

        Returns:
            Optional[Dict]: Контекст анализа
        """
        analysis_key = f"analysis:{analysis_id}"
        analysis_data = self.retrieve(analysis_key)

        if analysis_data:
            # Проверка целостности данных
            stored_hash = analysis_data.get("data_hash")
            current_hash = self._calculate_data_hash(analysis_data.get("context", {}))

            if stored_hash == current_hash:
                return analysis_data.get("context")
            else:
                logger.warning(f"ShortTermMemory: Нарушена целостность данных для анализа {analysis_id}")

        return None

    def cleanup_expired(self) -> int:
        """
        Очистка просроченных данных

        Returns:
            int: Количество очищенных записей
        """
        # Redis автоматически удаляет данные с истекшим TTL
        # Эта функция для мониторинга и отчетности
        try:
            # Получаем все ключи
            keys = self.redis_client.keys("stm:*")
            total = len(keys)

            # Проверяем TTL для каждого ключа
            expired_count = 0
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl < 0:  # -1 = нет TTL, -2 = ключ не существует
                    expired_count += 1

            if expired_count > 0:
                logger.info(f"ShortTermMemory: Найдено {expired_count} просроченных записей")

            return expired_count

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка очистки просроченных данных: {e}")
            return 0

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Получение статистики использования памяти

        Returns:
            Dict: Статистика использования
        """
        if not self.redis_client:
            return {"status": "redis_not_available"}

        try:
            # Получаем информацию из Redis
            info = self.redis_client.info()

            # Получаем ключи нашей памяти
            stm_keys = self.redis_client.keys("stm:*")
            session_keys = self.redis_client.keys("session:*")
            conversation_keys = self.redis_client.keys("conversation:*")
            analysis_keys = self.redis_client.keys("analysis:*")

            stats = {
                "status": "active",
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "stm_keys": len(stm_keys),
                "session_keys": len(session_keys),
                "conversation_keys": len(conversation_keys),
                "analysis_keys": len(analysis_keys),
                "uptime_days": info.get("uptime_in_days", 0),
                "connected_clients": info.get("connected_clients", 0),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                "timestamp": datetime.now().isoformat()
            }

            return stats

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка получения статистики: {e}")
            return {"status": "error", "error": str(e)}

    def _update_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Обновление метаданных для ключа"""
        if not self.redis_client:
            return

        try:
            # Получаем текущие данные
            serialized_data = self.redis_client.get(f"stm:{key}")
            if serialized_data:
                stored_data = pickle.loads(serialized_data)
                stored_data["metadata"] = metadata

                # Сохраняем обновленные данные
                updated_data = pickle.dumps(stored_data)
                ttl = self.redis_client.ttl(f"stm:{key}")

                if ttl > 0:
                    self.redis_client.setex(f"stm:{key}", ttl, updated_data)
                else:
                    self.redis_client.set(f"stm:{key}", updated_data)

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка обновления метаданных: {e}")

    def _calculate_data_hash(self, data: Any) -> str:
        """Расчет хэша данных для проверки целостности"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except:
            return ""

    def backup_to_file(self, filepath: str) -> bool:
        """
        Резервное копирование данных в файл

        Args:
            filepath: Путь к файлу для сохранения

        Returns:
            bool: Успешно ли сохранено
        """
        try:
            # Получаем все данные
            all_data = {}
            keys = self.redis_client.keys("stm:*")

            for key in keys:
                key_str = key.decode()
                data = self.retrieve(key_str.replace("stm:", ""))
                if data:
                    all_data[key_str] = data

            # Сохраняем в файл
            with open(filepath, 'wb') as f:
                pickle.dump(all_data, f)

            logger.info(f"ShortTermMemory: Резервная копия сохранена в {filepath}")
            return True

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка резервного копирования: {e}")
            return False

    def restore_from_file(self, filepath: str) -> bool:
        """
        Восстановление данных из файла

        Args:
            filepath: Путь к файлу с резервной копией

        Returns:
            bool: Успешно ли восстановлено
        """
        try:
            # Загружаем данные из файла
            with open(filepath, 'rb') as f:
                all_data = pickle.load(f)

            # Восстанавливаем данные
            for key, data in all_data.items():
                key_str = key.replace("stm:", "")
                self.store(key_str, data)

            logger.info(f"ShortTermMemory: Данные восстановлены из {filepath}")
            return True

        except Exception as e:
            logger.error(f"ShortTermMemory: Ошибка восстановления: {e}")
            return False