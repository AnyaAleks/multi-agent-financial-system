"""
Финансовый граф знаний
"""
import networkx as nx
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialEntity:
    """Финансовая сущность"""

    def __init__(self, entity_id: str, entity_type: str, name: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.entity_id = entity_id
        self.entity_type = entity_type  # COMPANY, PERSON, METRIC, EVENT, SECTOR
        self.name = name
        self.metadata = metadata or {}
        self.embedding = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def update_metadata(self, new_metadata: Dict[str, Any]):
        self.metadata.update(new_metadata)
        self.updated_at = datetime.now()


class FinancialRelationship:
    """Отношение между финансовыми сущностями"""

    def __init__(self, relationship_id: str, source_id: str, target_id: str,
                 relationship_type: str, properties: Optional[Dict[str, Any]] = None):
        self.relationship_id = relationship_id
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.properties = properties or {}
        self.confidence = 1.0
        self.sources: List[str] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "sources": self.sources,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def add_source(self, source: str):
        if source not in self.sources:
            self.sources.append(source)
        self.updated_at = datetime.now()


class FinancialKnowledgeGraph:
    """Финансовый граф знаний"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, FinancialEntity] = {}
        self.relationships: Dict[str, FinancialRelationship] = {}
        self.embedding_model = None
        self.neo4j_driver = None

        self._initialize_embedding_model()
        self._connect_neo4j()

    def _initialize_embedding_model(self):
        """Инициализация модели эмбеддингов"""
        try:
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Легкая модель для быстрых эмбеддингов
                device='cpu'
            )
            logger.info("KnowledgeGraph: Модель эмбеддингов инициализирована")
        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка инициализации модели эмбеддингов: {e}")

    def _connect_neo4j(self):
        """Подключение к Neo4j"""
        try:
            neo4j_url = settings.get("NEO4J_URL", "bolt://localhost:7687")
            neo4j_user = settings.get("NEO4J_USER", "neo4j")
            neo4j_password = settings.get("NEO4J_PASSWORD", "password")

            self.neo4j_driver = GraphDatabase.driver(
                neo4j_url,
                auth=(neo4j_user, neo4j_password)
            )

            # Тест подключения
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")

            logger.info("KnowledgeGraph: Подключено к Neo4j")

        except Exception as e:
            logger.warning(f"KnowledgeGraph: Не удалось подключиться к Neo4j: {e}")
            self.neo4j_driver = None

    def add_entity(self, entity: FinancialEntity) -> bool:
        """
        Добавление сущности в граф

        Args:
            entity: Финансовая сущность

        Returns:
            bool: Успешно ли добавлено
        """
        try:
            # Генерация эмбеддинга
            if self.embedding_model and not entity.embedding:
                entity.embedding = self._generate_embedding(entity.name)

            # Добавление в память
            self.entities[entity.entity_id] = entity
            self.graph.add_node(
                entity.entity_id,
                **entity.to_dict()
            )

            # Синхронизация с Neo4j
            self._sync_entity_to_neo4j(entity)

            logger.debug(f"KnowledgeGraph: Добавлена сущность: {entity.entity_id}")
            return True

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка добавления сущности: {e}")
            return False

    def add_relationship(self, relationship: FinancialRelationship) -> bool:
        """
        Добавление отношения в граф

        Args:
            relationship: Отношение между сущностями

        Returns:
            bool: Успешно ли добавлено
        """
        try:
            # Проверка существования сущностей
            if (relationship.source_id not in self.entities or
                    relationship.target_id not in self.entities):
                logger.warning(f"KnowledgeGraph: Сущности не найдены для отношения {relationship.relationship_id}")
                return False

            # Добавление в память
            self.relationships[relationship.relationship_id] = relationship
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                key=relationship.relationship_id,
                **relationship.to_dict()
            )

            # Синхронизация с Neo4j
            self._sync_relationship_to_neo4j(relationship)

            logger.debug(f"KnowledgeGraph: Добавлено отношение: {relationship.relationship_id}")
            return True

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка добавления отношения: {e}")
            return False

    def find_entity(self, query: str, entity_type: Optional[str] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск сущностей

        Args:
            query: Текст для поиска
            entity_type: Фильтр по типу сущности
            limit: Максимальное количество результатов

        Returns:
            List[Dict]: Найденные сущности
        """
        results = []

        try:
            # Поиск по имени
            query_lower = query.lower()
            for entity_id, entity in self.entities.items():
                if entity_type and entity.entity_type != entity_type:
                    continue

                if query_lower in entity.name.lower():
                    results.append({
                        "entity": entity.to_dict(),
                        "score": 1.0,
                        "match_type": "exact_name"
                    })

            # Семантический поиск с эмбеддингами
            if self.embedding_model and len(results) < limit:
                query_embedding = self._generate_embedding(query)
                semantic_results = []

                for entity_id, entity in self.entities.items():
                    if entity_type and entity.entity_type != entity_type:
                        continue

                    if entity.embedding is not None:
                        similarity = self._calculate_cosine_similarity(
                            query_embedding, entity.embedding
                        )

                        if similarity > 0.5:  # Порог схожести
                            semantic_results.append({
                                "entity": entity.to_dict(),
                                "score": similarity,
                                "match_type": "semantic"
                            })

                # Сортировка по схожести
                semantic_results.sort(key=lambda x: x["score"], reverse=True)
                results.extend(semantic_results[:limit - len(results)])

            # Сортировка и ограничение
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка поиска сущностей: {e}")
            return []

    def find_relationships(self, source_id: Optional[str] = None,
                           target_id: Optional[str] = None,
                           relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Поиск отношений

        Args:
            source_id: ID исходной сущности
            target_id: ID целевой сущности
            relationship_type: Тип отношения

        Returns:
            List[Dict]: Найденные отношения
        """
        results = []

        try:
            # Поиск в графе NetworkX
            if source_id and target_id:
                # Конкретное отношение между двумя сущностями
                if self.graph.has_edge(source_id, target_id):
                    edges_data = self.graph.get_edge_data(source_id, target_id)
                    for rel_id, edge_data in edges_data.items():
                        if relationship_type and edge_data.get("relationship_type") != relationship_type:
                            continue
                        results.append(edge_data)

            elif source_id:
                # Все исходящие отношения от сущности
                for target, edge_data in self.graph[source_id].items():
                    for rel_id, data in edge_data.items():
                        if relationship_type and data.get("relationship_type") != relationship_type:
                            continue
                        results.append(data)

            elif target_id:
                # Все входящие отношения к сущности
                for source in self.graph.predecessors(target_id):
                    edge_data = self.graph.get_edge_data(source, target_id)
                    for rel_id, data in edge_data.items():
                        if relationship_type and data.get("relationship_type") != relationship_type:
                            continue
                        results.append(data)

            else:
                # Все отношения указанного типа
                for source, target, edge_data in self.graph.edges(data=True):
                    if relationship_type and edge_data.get("relationship_type") != relationship_type:
                        continue
                    results.append(edge_data)

            return results

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка поиска отношений: {e}")
            return []

    def get_entity_neighbors(self, entity_id: str, depth: int = 1,
                             relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получение соседей сущности

        Args:
            entity_id: ID сущности
            depth: Глубина обхода
            relationship_types: Фильтр по типам отношений

        Returns:
            Dict: Соседи сущности
        """
        try:
            if entity_id not in self.entities:
                return {"error": "Entity not found"}

            neighbors = {
                "entity": self.entities[entity_id].to_dict(),
                "neighbors": {},
                "paths": []
            }

            # Обход в ширину
            visited = {entity_id}
            queue = [(entity_id, 0, [entity_id])]  # (node, depth, path)

            while queue:
                current_id, current_depth, path = queue.pop(0)

                if current_depth >= depth:
                    continue

                # Исходящие отношения
                for target_id in self.graph.successors(current_id):
                    edge_data = self.graph.get_edge_data(current_id, target_id)

                    for rel_id, data in edge_data.items():
                        if relationship_types and data.get("relationship_type") not in relationship_types:
                            continue

                        if target_id not in visited:
                            visited.add(target_id)
                            new_path = path + [target_id]

                            # Сохранение пути
                            neighbors["paths"].append({
                                "path": new_path,
                                "relationships": [data],
                                "depth": current_depth + 1
                            })

                            # Добавление соседа
                            if target_id not in neighbors["neighbors"]:
                                neighbors["neighbors"][target_id] = {
                                    "entity": self.entities[target_id].to_dict(),
                                    "relationships": []
                                }

                            neighbors["neighbors"][target_id]["relationships"].append(data)

                            # Добавление в очередь для дальнейшего обхода
                            queue.append((target_id, current_depth + 1, new_path))

            return neighbors

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка получения соседей: {e}")
            return {"error": str(e)}

    def query_cypher(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Выполнение Cypher запроса

        Args:
            cypher_query: Cypher запрос
            parameters: Параметры запроса

        Returns:
            List[Dict]: Результаты запроса
        """
        if not self.neo4j_driver:
            logger.warning("KnowledgeGraph: Neo4j не доступен, запрос не выполнен")
            return []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                records = []

                for record in result:
                    records.append(dict(record))

                return records

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка выполнения Cypher запроса: {e}")
            return []

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечение сущностей из текста

        Args:
            text: Текст для анализа

        Returns:
            List[Dict]: Извлеченные сущности
        """
        # В реальной системе здесь используется NER модель
        # Для примера используем простые правила

        entities = []

        # Простые правила для демонстрации
        import re

        # Поиск тикеров акций (например, AAPL, MSFT)
        ticker_pattern = r'\b([A-Z]{2,5})\b'
        tickers = re.findall(ticker_pattern, text)

        for ticker in tickers:
            # Проверка, является ли это известным тикером
            if self._is_known_ticker(ticker):
                entities.append({
                    "text": ticker,
                    "type": "COMPANY",
                    "confidence": 0.9,
                    "entity_id": f"company_{ticker}"
                })

        # Поиск финансовых метрик
        metric_keywords = {
            "revenue": "REVENUE",
            "earnings": "EARNINGS",
            "profit": "PROFIT",
            "margin": "MARGIN",
            "growth": "GROWTH",
            "debt": "DEBT",
            "cash": "CASH",
            "dividend": "DIVIDEND"
        }

        for keyword, metric_type in metric_keywords.items():
            if keyword in text.lower():
                entities.append({
                    "text": keyword,
                    "type": "METRIC",
                    "confidence": 0.7,
                    "entity_id": f"metric_{keyword}"
                })

        # Поиск персон
        person_pattern = r'\b(Mr\.|Ms\.|Mrs\.|Dr\.)?\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        persons = re.findall(person_pattern, text)

        for title, first_name, last_name in persons:
            full_name = f"{first_name} {last_name}"
            entities.append({
                "text": full_name,
                "type": "PERSON",
                "confidence": 0.8,
                "entity_id": f"person_{first_name.lower()}_{last_name.lower()}"
            })

        return entities

    def extract_relationships_from_text(self, text: str,
                                        entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Извлечение отношений из текста

        Args:
            text: Текст для анализа
            entities: Извлеченные сущности

        Returns:
            List[Dict]: Извлеченные отношения
        """
        relationships = []

        # Простые правила для демонстрации
        text_lower = text.lower()

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:], start=i + 1):
                # Проверка наличия слов, указывающих на отношения
                if self._find_relationship_between_entities(text_lower, entity1, entity2):
                    relationship_type = self._determine_relationship_type(
                        text_lower, entity1, entity2
                    )

                    if relationship_type:
                        relationships.append({
                            "source_id": entity1["entity_id"],
                            "target_id": entity2["entity_id"],
                            "relationship_type": relationship_type,
                            "confidence": 0.7,
                            "text_evidence": text
                        })

        return relationships

    def build_from_documents(self, documents: List[Dict[str, Any]]):
        """
        Построение графа из документов

        Args:
            documents: Список документов
        """
        logger.info(f"KnowledgeGraph: Начинаю построение графа из {len(documents)} документов")

        total_entities = 0
        total_relationships = 0

        for doc in documents:
            try:
                text = doc.get("content", "")
                source = doc.get("source", "unknown")

                # Извлечение сущностей
                entities = self.extract_entities_from_text(text)

                # Добавление сущностей в граф
                for entity_info in entities:
                    entity_id = entity_info["entity_id"]

                    if entity_id not in self.entities:
                        entity = FinancialEntity(
                            entity_id=entity_id,
                            entity_type=entity_info["type"],
                            name=entity_info["text"],
                            metadata={
                                "confidence": entity_info["confidence"],
                                "first_seen_in": source
                            }
                        )

                        if self.add_entity(entity):
                            total_entities += 1

                    # Обновление метаданных существующей сущности
                    else:
                        self.entities[entity_id].metadata.update({
                            "last_seen_in": source,
                            "mention_count": self.entities[entity_id].metadata.get("mention_count", 0) + 1
                        })

                # Извлечение и добавление отношений
                relationships = self.extract_relationships_from_text(text, entities)

                for rel_info in relationships:
                    relationship_id = f"rel_{hash(frozenset([rel_info['source_id'], rel_info['target_id'], rel_info['relationship_type']]))}"

                    if relationship_id not in self.relationships:
                        relationship = FinancialRelationship(
                            relationship_id=relationship_id,
                            source_id=rel_info["source_id"],
                            target_id=rel_info["target_id"],
                            relationship_type=rel_info["relationship_type"],
                            properties={
                                "confidence": rel_info["confidence"],
                                "evidence": rel_info.get("text_evidence", "")
                            }
                        )

                        relationship.add_source(source)

                        if self.add_relationship(relationship):
                            total_relationships += 1

                    # Обновление существующего отношения
                    else:
                        self.relationships[relationship_id].add_source(source)
                        self.relationships[relationship_id].confidence = max(
                            self.relationships[relationship_id].confidence,
                            rel_info["confidence"]
                        )

            except Exception as e:
                logger.error(f"KnowledgeGraph: Ошибка обработки документа: {e}")

        logger.info(
            f"KnowledgeGraph: Построение завершено. Добавлено {total_entities} сущностей и {total_relationships} отношений")

    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """
        Экспорт графа

        Args:
            format: Формат экспорта (json, graphml, etc.)

        Returns:
            Dict: Экспортированный граф
        """
        try:
            if format == "json":
                return {
                    "entities": [e.to_dict() for e in self.entities.values()],
                    "relationships": [r.to_dict() for r in self.relationships.values()],
                    "metadata": {
                        "total_entities": len(self.entities),
                        "total_relationships": len(self.relationships),
                        "exported_at": datetime.now().isoformat(),
                        "format": "json"
                    }
                }

            elif format == "graphml":
                # Экспорт в GraphML формат
                import io
                from networkx.readwrite import graphml

                output = io.StringIO()
                graphml.write_graphml(self.graph, output)

                return {
                    "content": output.getvalue(),
                    "format": "graphml",
                    "exported_at": datetime.now().isoformat()
                }

            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка экспорта графа: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики графа

        Returns:
            Dict: Статистика графа
        """
        try:
            # Основные метрики
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()

            # Распределение по типам сущностей
            entity_type_counts = {}
            for entity in self.entities.values():
                entity_type_counts[entity.entity_type] = entity_type_counts.get(entity.entity_type, 0) + 1

            # Распределение по типам отношений
            relationship_type_counts = {}
            for relationship in self.relationships.values():
                relationship_type_counts[relationship.relationship_type] = \
                    relationship_type_counts.get(relationship.relationship_type, 0) + 1

            # Плотность графа
            density = nx.density(self.graph) if total_nodes > 1 else 0

            # Компоненты связности
            connected_components = list(nx.weakly_connected_components(self.graph))

            # Средняя степень
            if total_nodes > 0:
                avg_degree = total_edges / total_nodes
            else:
                avg_degree = 0

            return {
                "total_entities": total_nodes,
                "total_relationships": total_edges,
                "entity_type_distribution": entity_type_counts,
                "relationship_type_distribution": relationship_type_counts,
                "graph_density": density,
                "connected_components": len(connected_components),
                "largest_component_size": max([len(c) for c in connected_components], default=0),
                "average_degree": avg_degree,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"KnowledgeGraph: Ошибка получения статистики: {e}")
            return {"error": str(e)}

    def _generate_embedding(self, text: str) -> List[float]:
        """Генерация эмбеддинга для текста"""
        try:
            if self.embedding_model:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            return []
        except:
            return []

    def _calculate_cosine_similarity(self, embedding1: List[float],
                                     embedding2: List[float]) -> float:
        """Расчет косинусной схожести"""
        try:
            import numpy as np

            if not embedding1 or not embedding2:
                return 0.0

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except:
            return 0.0

    def _sync_entity_to_neo4j(self, entity: FinancialEntity):
        """Синхронизация сущности с Neo4j"""
        if not self.neo4j_driver:
            return

        try:
            cypher = """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.entity_type = $entity_type,
                e.name = $name,
                e.metadata = $metadata,
                e.created_at = $created_at,
                e.updated_at = $updated_at
            """

            with self.neo4j_driver.session() as session:
                session.run(cypher, {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "metadata": entity.metadata,
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat()
                })

        except Exception as e:
            logger.warning(f"KnowledgeGraph: Ошибка синхронизации сущности с Neo4j: {e}")

    def _sync_relationship_to_neo4j(self, relationship: FinancialRelationship):
        """Синхронизация отношения с Neo4j"""
        if not self.neo4j_driver:
            return

        try:
            cypher = """
            MATCH (source:Entity {entity_id: $source_id})
            MATCH (target:Entity {entity_id: $target_id})
            MERGE (source)-[r:RELATIONSHIP {relationship_id: $relationship_id}]->(target)
            SET r.relationship_type = $relationship_type,
                r.properties = $properties,
                r.confidence = $confidence,
                r.sources = $sources,
                r.created_at = $created_at,
                r.updated_at = $updated_at
            """

            with self.neo4j_driver.session() as session:
                session.run(cypher, {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "relationship_id": relationship.relationship_id,
                    "relationship_type": relationship.relationship_type,
                    "properties": relationship.properties,
                    "confidence": relationship.confidence,
                    "sources": relationship.sources,
                    "created_at": relationship.created_at.isoformat(),
                    "updated_at": relationship.updated_at.isoformat()
                })

        except Exception as e:
            logger.warning(f"KnowledgeGraph: Ошибка синхронизации отношения с Neo4j: {e}")

    def _is_known_ticker(self, ticker: str) -> bool:
        """Проверка, является ли строка известным тикером"""
        known_tickers = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
            "WMT", "PG", "MA", "HD", "BAC", "DIS", "ADBE", "NFLX", "CRM", "PYPL"
        }
        return ticker in known_tickers

    def _find_relationship_between_entities(self, text: str,
                                            entity1: Dict[str, Any],
                                            entity2: Dict[str, Any]) -> bool:
        """Поиск отношения между сущностями в тексте"""
        # Простая проверка: сущности упоминаются близко друг к другу
        words = text.split()

        try:
            idx1 = words.index(entity1["text"].lower())
            idx2 = words.index(entity2["text"].lower())

            # Если сущности упоминаются в пределах 5 слов друг от друга
            return abs(idx1 - idx2) <= 5
        except ValueError:
            return False

    def _determine_relationship_type(self, text: str,
                                     entity1: Dict[str, Any],
                                     entity2: Dict[str, Any]) -> Optional[str]:
        """Определение типа отношения между сущностями"""
        # Простые правила для демонстрации
        relationship_keywords = {
            "reports": "REPORTS",
            "owns": "OWNS",
            "competes_with": "COMPETES_WITH",
            "partners_with": "PARTNERS_WITH",
            "acquired": "ACQUIRED",
            "invested_in": "INVESTED_IN",
            "employs": "EMPLOYS",
            "influences": "INFLUENCES"
        }

        for keyword, rel_type in relationship_keywords.items():
            if keyword in text:
                return rel_type

        return None