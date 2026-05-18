from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import ollama

from extractor import extract_statement
from limpeza import limpar_descricao
from rag_memoria import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_RAG_CSV_PATH,
    DEFAULT_RAG_DB_PATH,
    DEFAULT_RAG_EMBEDDING_MODEL,
    RagExample,
    RagMemoryStore,
)
from regras_classificacao import classificar_por_regras, remover_acentos


DEFAULT_MODEL = "llama3"
DEFAULT_CATEGORIES = [
    "Supermercado",
    "Restauracao",
    "Transportes",
    "Saude",
    "Casa",
    "Tecnologia",
    "Subscricoes",
    "Transferencias",
    "Rendimentos",
    "Impostos",
    "Taxas",
    "Levantamentos",
    "Compras",
    "Outros",
]


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Extrai movimentos de um PDF e classifica-os com Ollama."
    )
    argument_parser.add_argument("pdf_path", help="Path para o extrato PDF.")
    argument_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Ollama a usar. Default: {DEFAULT_MODEL}.",
    )
    argument_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Formato de saida. Default: json.",
    )
    argument_parser.add_argument(
        "--output",
        help="Ficheiro de saida opcional. Se omitido, escreve no stdout.",
    )
    argument_parser.add_argument(
        "--categorias",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Lista de categorias permitidas.",
    )
    argument_parser.add_argument(
        "--sem-ia",
        action="store_true",
        help="Nao usa Ollama. Aplica apenas regras; o resto fica em Outros.",
    )
    argument_parser.add_argument(
        "--usar-rag",
        action="store_true",
        help="Consulta memoria local em Chroma antes do fallback para LLM.",
    )
    argument_parser.add_argument(
        "--rag-csv",
        default=DEFAULT_RAG_CSV_PATH,
        help=f"CSV usado para popular a memoria RAG. Default: {DEFAULT_RAG_CSV_PATH}.",
    )
    argument_parser.add_argument(
        "--rag-db-path",
        default=DEFAULT_RAG_DB_PATH,
        help=f"Pasta do Chroma persistente. Default: {DEFAULT_RAG_DB_PATH}.",
    )
    argument_parser.add_argument(
        "--rag-collection",
        default=DEFAULT_RAG_COLLECTION,
        help=f"Colecao Chroma da memoria. Default: {DEFAULT_RAG_COLLECTION}.",
    )
    argument_parser.add_argument(
        "--rag-embedding-model",
        default=DEFAULT_RAG_EMBEDDING_MODEL,
        help=f"Modelo Ollama usado para embeddings do RAG. Default: {DEFAULT_RAG_EMBEDDING_MODEL}.",
    )
    argument_parser.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Numero de exemplos RAG a recuperar. Default: 3.",
    )
    return argument_parser


def normalizar_categoria(raw_category: str, allowed_categories: list[str]) -> str:
    normalized = raw_category.strip().splitlines()[0].strip(" .,:;")
    normalized_ascii = remover_acentos(normalized).casefold()

    for category in allowed_categories:
        if normalized_ascii == remover_acentos(category).casefold():
            return category

    for category in allowed_categories:
        category_ascii = remover_acentos(category).casefold()
        if category_ascii in normalized_ascii:
            return category

    return normalized or "Outros"


def formatar_exemplos_rag(examples: list[RagExample]) -> str:
    if not examples:
        return "Sem exemplos RAG relevantes."

    lines = []
    for index, example in enumerate(examples[:3], start=1):
        distance_text = (
            f"{example.distance:.4f}" if example.distance is not None else "exact"
        )
        lines.append(
            f"Exemplo {index}: descricao={example.description} | "
            f"categoria={example.category} | distancia={distance_text}"
        )
    return "\n".join(lines)


def classificar_transacao(
    description: str,
    cleaned_description: str,
    details: str | None,
    direction: str,
    model: str,
    categories: list[str],
    rag_examples: list[RagExample] | None = None,
) -> tuple[str, str]:
    category_list = ", ".join(categories)
    rag_block = formatar_exemplos_rag(rag_examples or [])
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classificas transacoes bancarias. "
                    "Responde apenas com o nome exato de uma categoria da lista fornecida. "
                    "Usa os exemplos RAG como memoria historica quando fizer sentido. "
                    "Se nao houver categoria clara, responde Outros."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Categorias permitidas: {category_list}\n"
                    f"Descricao original: {description}\n"
                    f"Descricao limpa: {cleaned_description}\n"
                    f"Detalhes adicionais: {details or 'Sem detalhes'}\n"
                    f"Direcao do movimento: {direction}\n"
                    f"Exemplos RAG:\n{rag_block}\n"
                    "Devolve apenas uma categoria."
                ),
            },
        ],
    )
    raw_category = response["message"]["content"]
    source = "llm_rag" if rag_examples else "llm"
    return normalizar_categoria(raw_category, categories), source


def enriquecer_transacoes(
    pdf_path: str | Path,
    model: str,
    categories: list[str],
    sem_ia: bool = False,
    usar_rag: bool = False,
    rag_csv: str = DEFAULT_RAG_CSV_PATH,
    rag_db_path: str = DEFAULT_RAG_DB_PATH,
    rag_collection: str = DEFAULT_RAG_COLLECTION,
    rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    rag_top_k: int = 3,
) -> dict[str, object]:
    extraction = extract_statement(pdf_path)
    cache: dict[tuple[str, str], tuple[str, str]] = {}
    enriched_transactions: list[dict[str, str | None]] = []
    rag_store: RagMemoryStore | None = None

    if usar_rag:
        rag_store = RagMemoryStore(
            db_path=rag_db_path,
            collection_name=rag_collection,
            embedding_model=rag_embedding_model,
        )
        rag_store.sync_from_csv(rag_csv)

    for transaction in extraction.transactions:
        transaction_record = transaction.to_record()
        cleaned_description = limpar_descricao(transaction.description)
        cache_key = (cleaned_description, transaction.direction)
        rag_examples: list[RagExample] = []

        if cache_key not in cache:
            rule_result = classificar_por_regras(
                description=transaction.description,
                cleaned_description=cleaned_description,
                details=transaction_record["details"],
                direction=transaction.direction,
            )
            if rule_result is not None:
                cache[cache_key] = rule_result
            elif rag_store is not None:
                rag_lookup = rag_store.classify_with_memory(
                    description=transaction.description,
                    clean_description=cleaned_description,
                    details=transaction_record["details"],
                    direction=transaction.direction,
                    bank_name=transaction.bank_name if hasattr(transaction, "bank_name") else "",
                    amount=str(transaction.amount),
                    top_k=rag_top_k,
                )
                rag_examples = rag_lookup.examples
                if rag_lookup.category and rag_lookup.source:
                    cache[cache_key] = (rag_lookup.category, rag_lookup.source)
                elif sem_ia:
                    cache[cache_key] = ("Outros", "sem_ia")
                else:
                    cache[cache_key] = classificar_transacao(
                        description=transaction.description,
                        cleaned_description=cleaned_description,
                        details=transaction_record["details"],
                        direction=transaction.direction,
                        model=model,
                        categories=categories,
                        rag_examples=rag_examples,
                    )
            elif sem_ia:
                cache[cache_key] = ("Outros", "sem_ia")
            else:
                cache[cache_key] = classificar_transacao(
                    description=transaction.description,
                    cleaned_description=cleaned_description,
                    details=transaction_record["details"],
                    direction=transaction.direction,
                    model=model,
                    categories=categories,
                )

        transaction_record["clean_description"] = cleaned_description
        transaction_record["category"] = cache[cache_key][0]
        transaction_record["classification_source"] = cache[cache_key][1]
        enriched_transactions.append(transaction_record)

    return {
        "metadata": extraction.metadata.to_record(),
        "classification": {
            "model": model,
            "categories": categories,
            "sem_ia": sem_ia,
            "usar_rag": usar_rag,
            "rag_collection": rag_collection if usar_rag else None,
            "rag_embedding_model": rag_embedding_model if usar_rag else None,
        },
        "transactions": enriched_transactions,
    }


def write_json(payload: dict[str, object], output_path: str | None) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
        return
    print(content)


def write_csv(records: list[dict[str, object]], output_path: str | None) -> None:
    if not records:
        fieldnames = [
            "bank_name",
            "source_file",
            "page_number",
            "booking_date",
            "value_date",
            "description",
            "clean_description",
            "category",
            "classification_source",
            "amount",
            "direction",
            "balance",
            "currency",
            "details",
            "raw_text",
        ]
    else:
        fieldnames = list(records[0].keys())

    if output_path:
        output_stream = Path(output_path).open("w", newline="", encoding="utf-8")
        should_close = True
    else:
        output_stream = sys.stdout
        should_close = False

    try:
        writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    finally:
        if should_close:
            output_stream.close()


def main() -> None:
    args = build_argument_parser().parse_args()
    payload = enriquecer_transacoes(
        pdf_path=args.pdf_path,
        model=args.model,
        categories=args.categorias,
        sem_ia=args.sem_ia,
        usar_rag=args.usar_rag,
        rag_csv=args.rag_csv,
        rag_db_path=args.rag_db_path,
        rag_collection=args.rag_collection,
        rag_embedding_model=args.rag_embedding_model,
        rag_top_k=args.rag_top_k,
    )

    if args.format == "csv":
        write_csv(payload["transactions"], args.output)
        return

    write_json(payload, args.output)


if __name__ == "__main__":
    main()
