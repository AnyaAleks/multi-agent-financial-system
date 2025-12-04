"""
Гибридная система извлечения информации
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.rag.knowledge_graph import FinancialKnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Гибридный извлекатель информации"""

    def __init__(self, knowledge_graph: FinancialKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.vector_index = None
        self.bm25_index = None
        self.cross_encoder = None
        self.documents: List[Dict[str, Any]] = []
        self.document_embeddings: List[List[float]] = []

        self._initialize_models()

    def _initialize_models(self):
        """Инициализация моделей"""
        try:
            # Инициализация cross-encoder для реранкинга
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Retriever: Модели инициализированы")
        except Exception as e:
            logger.error(f"Retriever: Ошибка инициализации моделей: {e}")

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Индексация документов

        Args:
            documents: Список документов для индексации
        """
        logger.info(f"Retriever: Индексация {len(documents)} документов")

        self.documents = documents

        # Построение BM25 индекса
        tokenized_docs = [self._tokenize(doc.get("content", "")) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

        # Генерация векторных эмбеддингов
        self._generate_document_embeddings()

        logger.info("Retriever: Индексация завершена")

    def retrieve(self, query: str, query_type: str = "factual",
                top_k: int = 10, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Извлечение релевантных документов

        Args:
            query: Поисковый запрос
            query_type: Тип запроса (factual, comparative, causal, complex)
            top_k: Количество возвращаемых документов
            use_reranking: Использовать ли реранкинг

        Returns:
            List[Dict]: Релевантные документы
        """
        logger.info(f"Retriever: Извлечение для запроса: {query} (тип: {query_type})")

        try:
            # Классификация запроса
            query_category = self._classify_query(query, query_type)

            # Выбор стратегии извлечения
            retrieval_strategy = self._select_retrieval_strategy(query_category)

            # Извлечение документов
            candidates = self._execute_retrieval_strategy(
                query, retrieval_strategy, query_category
            )

            if not candidates:
                logger.warning("Retriever: Не найдено кандидатов")
                return []

            # Реранкинг результатов
            if use_reranking and self.cross_encoder:
                candidates = self._rerank_results(query, candidates)

            # Ограничение количества результатов
            final_results = candidates[:top_k]

            # Добавление метаданных
            for result in final_results:
                result["retrieval_strategy"] = retrieval_strategy
                result["query_category"] = query_category
                result["retrieved_at"] = datetime.now().isoformat()

            logger.info(f"Retriever: Найдено {len(final_results)} релевантных документов")
            return final_results

        except Exception as e:
            logger.error(f"Retriever: Ошибка извлечения: {e}")
            return []

    def _classify_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Классификация запроса

        Args:
            query: Поисковый запрос
            query_type: Заданный тип запроса

        Returns:
            Dict: Категория и параметры запроса
        """
        # Простая классификация на основе ключевых слов
        query_lower = query.lower()

        category = {
            "type": query_type,
            "entities": [],
            "intent": "unknown",
            "complexity": "simple"
        }

        # Извлечение сущностей из запроса
        entities = self.knowledge_graph.extract_entities_from_text(query)
        category["entities"] = entities

        # Определение намерения
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            category["intent"] = "comparison"
            category["complexity"] = "medium"
        elif any(word in query_lower for word in ["why", "cause", "reason", "because"]):
            category["intent"] = "causal"
            category["complexity"] = "medium"
        elif any(word in query_lower for word in ["how", "affect", "impact", "influence"]):
            category["intent"] = "explanatory"
            category["complexity"] = "complex"
        elif "?" in query:
            category["intent"] = "question"
        else:
            category["intent"] = "factual"

        # Определение сложности
        if len(entities) > 2:
            category["complexity"] = "complex"
        elif len(entities) == 2:
            category["complexity"] = "medium"

        return category

    def _select_retrieval_strategy(self, query_category: Dict[str, Any]) -> str:
        """
        Выбор стратегии извлечения

        Args:
            query_category: Категория запроса

        Returns:
            str: Выбранная стратегия
        """
        query_type = query_category["type"]
        intent = query_category["intent"]
        complexity = query_category["complexity"]

        if query_type == "factual" or intent == "factual":
            return "knowledge_graph"
        elif intent == "comparison":
            return "multi_query_vector"
        elif intent in ["causal", "explanatory"]:
            return "temporal_vector"
        elif complexity == "complex":
            return "graph_reasoning"
        else:
            return "hybrid_search"

    def _execute_retrieval_strategy(self, query: str, strategy: str,
                                   query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Выполнение стратегии извлечения

        Args:
            query: Поисковый запрос
            strategy: Стратегия извлечения
            query_category: Категория запроса

        Returns:
            List[Dict]: Кандидаты документы
        """
        candidates = []

        if strategy == "knowledge_graph":
            candidates = self._knowledge_graph_lookup(query, query_category)
        elif strategy == "multi_query_vector":
            candidates = self._multi_query_vector_search(query, query_category)
        elif strategy == "temporal_vector":
            candidates = self._temporal_vector_search(query, query_category)
        elif strategy == "graph_reasoning":
            candidates = self._graph_reasoning_search(query, query_category)
        else:  # hybrid_search
            candidates = self._hybrid_search(query, query_category)

        return candidates

    def _knowledge_graph_lookup(self, query: str,
                               query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск в графе знаний"""
        candidates = []

        try:
            entities = query_category["entities"]

            for entity in entities:
                # Поиск сущности в графе
                found_entities = self.knowledge_graph.find_entity(
                    entity["text"],
                    entity["type"]
                )

                for found in found_entities:
                    # Получение соседей сущности
                    neighbors = self.knowledge_graph.get_entity_neighbors(
                        found["entity"]["entity_id"],
                        depth=2
                    )

                    # Создание документа из информации о сущности
                    doc = self._create_document_from_entity(found["entity"], neighbors)
                    doc["score"] = found["score"]
                    doc["source"] = "knowledge_graph"

                    candidates.append(doc)

            # Сортировка по score
            candidates.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Retriever: Ошибка поиска в графе знаний: {e}")

        return candidates

    def _multi_query_vector_search(self, query: str,
                                  query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Многозапросный векторный поиск"""
        candidates = []

        try:
            entities = query_category["entities"]

            # Создание отдельных запросов для каждой сущности
            sub_queries = []
            for entity in entities:
                sub_query = f"{entity['text']} {query}"
                sub_queries.append(sub_query)

            # Если есть только одна сущность, добавляем общий контекст
            if len(sub_queries) == 1:
                sub_queries.append(query + " comparison analysis")

            # Выполнение поиска для каждого подзапроса
            all_candidates = []
            for sub_query in sub_queries:
                sub_candidates = self._vector_search(sub_query, top_k=20)
                all_candidates.extend(sub_candidates)

            # Дедупликация
            seen_ids = set()
            for candidate in all_candidates:
                doc_id = candidate.get("doc_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidates.append(candidate)

            # Обновление score для сравнений
            for candidate in candidates:
                candidate["score"] *= 1.2  # Буст для сравнительных запросов
                candidate["source"] = "multi_query_vector"

            # Сортировка
            candidates.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Retriever: Ошибка многозапросного поиска: {e}")

        return candidates

    def _temporal_vector_search(self, query: str,
                               query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Временной векторный поиск"""
        candidates = []

        try:
            # Извлечение временных меток из запроса
            time_keywords = {
                "recently": -7,  # 7 дней назад
                "last week": -7,
                "last month": -30,
                "yesterday": -1,
                "today": 0,
                "2024": 0,
                "2023": -365
            }

            time_filter = None
            query_lower = query.lower()

            for keyword, days_offset in time_keywords.items():
                if keyword in query_lower:
                    time_filter = datetime.now().timestamp() + (days_offset * 86400)
                    break

            # Векторный поиск
            vector_candidates = self._vector_search(query, top_k=50)

            # Фильтрация по времени
            for candidate in vector_candidates:
                doc_timestamp = candidate.get("metadata", {}).get("timestamp")

                if doc_timestamp and time_filter:
                    try:
                        doc_time = datetime.fromisoformat(doc_timestamp).timestamp()
                        if doc_time >= time_filter:
                            candidate["score"] *= 1.3  # Буст для релевантных по времени
                            candidate["source"] = "temporal_vector"
                            candidates.append(candidate)
                    except:
                        pass
                else:
                    candidate["source"] = "temporal_vector"
                    candidates.append(candidate)

            # Сортировка
            candidates.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Retriever: Ошибка временного поиска: {e}")

        return candidates

    def _graph_reasoning_search(self, query: str,
                               query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Поиск с графовыми рассуждениями"""
        candidates = []

        try:
            entities = query_category["entities"]

            if len(entities) >= 2:
                # Поиск путей между сущностями
                entity1 = entities[0]
                entity2 = entities[1]

                # Поиск сущностей в графе
                found1 = self.knowledge_graph.find_entity(entity1["text"], entity1["type"])
                found2 = self.knowledge_graph.find_entity(entity2["text"], entity2["type"])

                if found1 and found2:
                    entity1_id = found1[0]["entity"]["entity_id"]
                    entity2_id = found2[0]["entity"]["entity_id"]

                    # Поиск путей между сущностями
                    # (В реальной системе здесь используется алгоритм поиска путей)

                    # Временная реализация: поиск документов, содержащих обе сущности
                    for doc in self.documents:
                        content = doc.get("content", "").lower()
                        if (entity1["text"].lower() in content and
                            entity2["text"].lower() in content):

                            # Создание кандидата
                            candidate = {
                                "doc_id": doc.get("doc_id", hash(content)),
                                "content": content[:500] + "...",
                                "metadata": doc.get("metadata", {}),
                                "score": 0.8,
                                "source": "graph_reasoning",
                                "reasoning_path": f"{entity1['text']} -> {entity2['text']}"
                            }
                            candidates.append(candidate)

            # Если не нашли через граф, используем гибридный поиск
            if not candidates:
                candidates = self._hybrid_search(query, query_category)
                for candidate in candidates:
                    candidate["source"] = "graph_reasoning_fallback"

        except Exception as e:
            logger.error(f"Retriever: Ошибка графового поиска: {e}")

        return candidates

    def _hybrid_search(self, query: str,
                      query_category: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Гибридный поиск (векторный + ключевые слова)"""
        candidates = []

        try:
            # Векторный поиск
            vector_candidates = self._vector_search(query, top_k=30)

            # Поиск по ключевым словам (BM25)
            bm25_candidates = self._bm25_search(query, top_k=30)

            # Объединение результатов
            all_candidates = {}

            for candidate in vector_candidates:
                doc_id = candidate.get("doc_id")
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        "candidate": candidate,
                        "scores": {"vector": candidate["score"]}
                    }
                else:
                    all_candidates[doc_id]["scores"]["vector"] = candidate["score"]

            for candidate in bm25_candidates:
                doc_id = candidate.get("doc_id")
                if doc_id in all_candidates:
                    all_candidates[doc_id]["scores"]["bm25"] = candidate["score"]
                else:
                    all_candidates[doc_id] = {
                        "candidate": candidate,
                        "scores": {"bm25": candidate["score"]}
                    }

            # Вычисление комбинированного score
            for doc_id, data in all_candidates.items():
                candidate = data["candidate"]
                scores = data["scores"]

                # Веса для разных методов
                vector_score = scores.get("vector", 0)
                bm25_score = scores.get("bm25", 0)

                # Комбинированный score (можно настроить веса)
                combined_score = (vector_score * 0.7) + (bm25_score * 0.3)

                candidate["score"] = combined_score
                candidate["source"] = "hybrid_search"
                candidate["score_breakdown"] = {
                    "vector": vector_score,
                    "bm25": bm25_score,
                    "combined": combined_score
                }

                candidates.append(candidate)

            # Сортировка
            candidates.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Retriever: Ошибка гибридного поиска: {e}")

        return candidates

    def _vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Векторный поиск"""
        candidates = []

        try:
            if not self.document_embeddings:
                return []

            # Генерация эмбеддинга запроса
            query_embedding = self._generate_query_embedding(query)

            # Расчет схожести
            similarities = []
            for i, doc_embedding in enumerate(self.document_embeddings):
                if doc_embedding:
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    similarities.append((i, similarity))

            # Сортировка по схожести
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Создание кандидатов
            for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    candidate = {
                        "doc_id": doc.get("doc_id", f"doc_{doc_idx}"),
                        "content": doc.get("content", "")[:500] + "...",
                        "metadata": doc.get("metadata", {}),
                        "score": float(similarity),
                        "rank": i + 1,
                        "search_method": "vector"
                    }
                    candidates.append(candidate)

        except Exception as e:
            logger.error(f"Retriever: Ошибка векторного поиска: {e}")

        return candidates

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Поиск по ключевым словам (BM25)"""
        candidates = []

        try:
            if not self.bm25_index:
                return []

            # Токенизация запроса
            tokenized_query = self._tokenize(query)

            # Получение scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Получение топ-K индексов
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Создание кандидатов
            for i, doc_idx in enumerate(top_indices):
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    score = scores[doc_idx]

                    if score > 0:  # Только релевантные документы
                        candidate = {
                            "doc_id": doc.get("doc_id", f"doc_{doc_idx}"),
                            "content": doc.get("content", "")[:500] + "...",
                            "metadata": doc.get("metadata", {}),
                            "score": float(score),
                            "rank": i + 1,
                            "search_method": "bm25"
                        }
                        candidates.append(candidate)

        except Exception as e:
            logger.error(f"Retriever: Ошибка BM25 поиска: {e}")

        return candidates

    def _rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Реранкинг результатов с помощью cross-encoder"""
        try:
            if not candidates or not self.cross_encoder:
                return candidates

            # Подготовка пар запрос-документ
            query_doc_pairs = []
            for candidate in candidates:
                content = candidate.get("content", "")
                query_doc_pairs.append([query, content])

            # Предсказание релевантности
            similarity_scores = self.cross_encoder.predict(query_doc_pairs)

            # Обновление scores
            for i, candidate in enumerate(candidates):
                candidate["reranked_score"] = float(similarity_scores[i])
                candidate["final_score"] = (
                    candidate.get("score", 0) * 0.3 +
                    candidate["reranked_score"] * 0.7
                )

            # Сортировка по final_score
            candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            logger.debug(f"Retriever: Выполнен реранкинг {len(candidates)} документов")

        except Exception as e:
            logger.error(f"Retriever: Ошибка реранкинга: {e}")

        return candidates

    def _generate_document_embeddings(self):
        """Генерация эмбеддингов для документов"""
        try:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            texts = [doc.get("content", "") for doc in self.documents]
            self.document_embeddings = self._embedding_model.encode(texts).tolist()

        except Exception as e:
            logger.error(f"Retriever: Ошибка генерации эмбеддингов: {e}")
            self.document_embeddings = []

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Генерация эмбеддинга для запроса"""
        try:
            if hasattr(self, '_embedding_model'):
                return self._embedding_model.encode([query])[0].tolist()
            return []
        except:
            return []

    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        # Простая токенизация для демонстрации
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Расчет косинусной схожести"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def _create_document_from_entity(self, entity: Dict[str, Any],
                                   neighbors: Dict[str, Any]) -> Dict[str, Any]:
        """Создание документа из информации о сущности"""
        entity_info = f"Entity: {entity['name']} (Type: {entity['entity_type']})"

        neighbors_info = []
        for neighbor_id, neighbor_data in neighbors.get("neighbors", {}).items():
            neighbor_entity = neighbor_data["entity"]
            relationships = neighbor_data["relationships"]

            rel_info = ", ".join([r["relationship_type"] for r in relationships])
            neighbors_info.append(
                f"{neighbor_entity['name']} [{rel_info}]"
            )

        content = f"{entity_info}\n\nConnected to:\n" + "\n".join(neighbors_info[:5])

        return {
            "doc_id": f"kg_{entity['entity_id']}",
            "content": content,
            "metadata": {
                "entity_id": entity["entity_id"],
                "entity_type": entity["entity_type"],
                "source": "knowledge_graph",
                "timestamp": datetime.now().isoformat()
            },
            "score": 1.0
        }

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Получение статистики извлечения

        Returns:
            Dict: Статистика
        """
        return {
            "total_documents": len(self.documents),
            "has_vector_index": len(self.document_embeddings) > 0,
            "has_bm25_index": self.bm25_index is not None,
            "has_cross_encoder": self.cross_encoder is not None,
            "knowledge_graph_entities": len(self.knowledge_graph.entities),
            "timestamp": datetime.now().isoformat()
        }