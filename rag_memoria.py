from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from statistics import mean

import chromadb
import ollama

from limpeza import limpar_descricao


DEFAULT_RAG_CSV_PATH = "avaliacao_transacoes.csv"
DEFAULT_RAG_DB_PATH = "db/chroma"
DEFAULT_RAG_COLLECTION = "transaction_memory"
DEFAULT_RAG_EMBEDDING_MODEL = "llama3"


@dataclass(frozen=True)
class RagExample:
    category: str
    description: str
    clean_description: str
    direction: str
    source_file: str
    booking_date: str
    amount: str
    distance: float | None = None


@dataclass
class RagLookupResult:
    category: str | None
    source: str | None
    examples: list[RagExample]


def build_memory_text(
    description: str,
    clean_description: str,
    details: str | None,
    direction: str,
    bank_name: str = "",
    amount: str = "",
    category: str = "",
) -> str:
    parts = [
        f"descricao_original: {description}",
        f"descricao_limpa: {clean_description}",
        f"detalhes: {details or 'sem detalhes'}",
        f"direcao: {direction}",
    ]

    if bank_name:
        parts.append(f"banco: {bank_name}")
    if amount:
        parts.append(f"montante: {amount}")
    if category:
        parts.append(f"categoria: {category}")

    return "\n".join(parts)


def build_example_id(
    source_file: str,
    booking_date: str,
    description: str,
    amount: str,
    direction: str,
) -> str:
    raw = "||".join([source_file, booking_date, description, amount, direction])
    return sha1(raw.encode("utf-8")).hexdigest()


class RagMemoryStore:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_RAG_DB_PATH,
        collection_name: str = DEFAULT_RAG_COLLECTION,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(model=self.embedding_model, input=texts)
        return response.embeddings

    def sync_from_truth_csv(self, csv_path: str | Path) -> int:
        path = Path(csv_path)
        if not path.exists():
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            for row in reader:
                if (row.get("extractor_status") or "").strip() == "missing_in_current_extractor":
                    continue

                source_file = (row.get("source_file") or "").strip()
                booking_date = (row.get("booking_date") or "").strip()
                description = (row.get("description") or "").strip()
                amount = (row.get("amount") or "").strip()
                direction = (row.get("direction") or "").strip()
                category = (row.get("categoria_correta") or "").strip()
                bank_name = (row.get("bank_name") or "").strip()
                clean_description = limpar_descricao(description)

                ids.append(
                    build_example_id(
                        source_file=source_file,
                        booking_date=booking_date,
                        description=description,
                        amount=amount,
                        direction=direction,
                    )
                )
                documents.append(
                    build_memory_text(
                        description=description,
                        clean_description=clean_description,
                        details=row.get("observacoes") or "",
                        direction=direction,
                        bank_name=bank_name,
                        amount=amount,
                        category=category,
                    )
                )
                metadatas.append(
                    {
                        "source_file": source_file,
                        "bank_name": bank_name,
                        "booking_date": booking_date,
                        "description": description,
                        "clean_description": clean_description,
                        "amount": amount,
                        "direction": direction,
                        "category": category,
                    }
                )

        if not ids:
            return 0

        if self.collection.count() == len(ids):
            return 0

        embeddings = self.embed_texts(documents)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def exact_lookup(self, clean_description: str) -> list[RagExample]:
        result = self.collection.get(
            where={"clean_description": clean_description},
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        examples: list[RagExample] = []

        for metadata in metadatas:
            if not metadata:
                continue
            examples.append(
                RagExample(
                    category=str(metadata.get("category", "")),
                    description=str(metadata.get("description", "")),
                    clean_description=str(metadata.get("clean_description", "")),
                    direction=str(metadata.get("direction", "")),
                    source_file=str(metadata.get("source_file", "")),
                    booking_date=str(metadata.get("booking_date", "")),
                    amount=str(metadata.get("amount", "")),
                    distance=None,
                )
            )

        return examples

    def similarity_lookup(
        self,
        description: str,
        clean_description: str,
        details: str | None,
        direction: str,
        bank_name: str = "",
        amount: str = "",
        top_k: int = 3,
    ) -> list[RagExample]:
        query_text = build_memory_text(
            description=description,
            clean_description=clean_description,
            details=details,
            direction=direction,
            bank_name=bank_name,
            amount=amount,
        )
        query_embedding = self.embed_texts([query_text])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        metadata_rows = (result.get("metadatas") or [[]])[0]
        distance_rows = (result.get("distances") or [[]])[0]
        examples: list[RagExample] = []

        for metadata, distance in zip(metadata_rows, distance_rows):
            if not metadata:
                continue
            examples.append(
                RagExample(
                    category=str(metadata.get("category", "")),
                    description=str(metadata.get("description", "")),
                    clean_description=str(metadata.get("clean_description", "")),
                    direction=str(metadata.get("direction", "")),
                    source_file=str(metadata.get("source_file", "")),
                    booking_date=str(metadata.get("booking_date", "")),
                    amount=str(metadata.get("amount", "")),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return examples

    def classify_with_memory(
        self,
        description: str,
        clean_description: str,
        details: str | None,
        direction: str,
        bank_name: str = "",
        amount: str = "",
        top_k: int = 3,
    ) -> RagLookupResult:
        exact_examples = self.exact_lookup(clean_description)
        exact_categories = sorted({example.category for example in exact_examples if example.category})
        if len(exact_categories) == 1:
            return RagLookupResult(
                category=exact_categories[0],
                source="rag_memoria_exata",
                examples=exact_examples,
            )

        similar_examples = self.similarity_lookup(
            description=description,
            clean_description=clean_description,
            details=details,
            direction=direction,
            bank_name=bank_name,
            amount=amount,
            top_k=top_k,
        )
        if not similar_examples:
            return RagLookupResult(category=None, source=None, examples=[])

        top_example = similar_examples[0]
        if top_example.distance is not None and top_example.distance <= 0.03:
            return RagLookupResult(
                category=top_example.category,
                source="rag_memoria_semelhante",
                examples=similar_examples,
            )

        category_groups: dict[str, list[RagExample]] = {}
        for example in similar_examples:
            category_groups.setdefault(example.category, []).append(example)

        best_category = None
        best_group: list[RagExample] = []
        for category, group in category_groups.items():
            if len(group) > len(best_group):
                best_category = category
                best_group = group

        if best_category and len(best_group) >= 2:
            distances = [example.distance for example in best_group if example.distance is not None]
            average_distance = mean(distances) if distances else None
            if average_distance is not None and average_distance <= 0.10:
                return RagLookupResult(
                    category=best_category,
                    source="rag_memoria_maioria",
                    examples=similar_examples,
                )

        return RagLookupResult(category=None, source=None, examples=similar_examples)
