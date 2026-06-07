from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from processar import DEFAULT_CATEGORIES, DEFAULT_MODEL, enriquecer_transacoes
from rag_memoria import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_RAG_CSV_PATH,
    DEFAULT_RAG_DB_PATH,
    DEFAULT_RAG_EMBEDDING_MODEL,
)
from regras_classificacao import remover_acentos


@dataclass(frozen=True)
class TruthTransaction:
    source_file: str
    bank_name: str
    booking_date: str
    description: str
    amount: str
    direction: str
    categoria_correta: str
    extractor_status: str
    observacoes: str
    occurrence: int

    @property
    def key(self) -> tuple[str, str, str, str, str, int]:
        return (
            self.source_file,
            self.booking_date,
            self.description,
            self.amount,
            self.direction,
            self.occurrence,
        )


@dataclass(frozen=True)
class PredictedTransaction:
    source_file: str
    bank_name: str
    booking_date: str
    description: str
    amount: str
    direction: str
    category: str
    classification_source: str
    occurrence: int

    @property
    def key(self) -> tuple[str, str, str, str, str, int]:
        return (
            self.source_file,
            self.booking_date,
            self.description,
            self.amount,
            self.direction,
            self.occurrence,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Avalia o output do processar.py contra o avaliacao_transacoes.csv."
    )
    argument_parser.add_argument(
        "--csv-path",
        default="avaliacao_transacoes.csv",
        help="Path para o CSV de verdade de referencia.",
    )
    argument_parser.add_argument(
        "--source-file",
        help="Avalia apenas um ficheiro fonte do CSV.",
    )
    argument_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Ollama a usar. Default: {DEFAULT_MODEL}.",
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
        help="Avalia em modo sem IA, usando apenas regras.",
    )
    argument_parser.add_argument(
        "--usar-rag",
        action="store_true",
        help="Avalia o pipeline com memoria RAG ativa.",
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


def normalize_amount(value: str) -> str:
    cleaned = value.replace(",", ".").strip()
    try:
        return f"{Decimal(cleaned):.2f}"
    except InvalidOperation:
        return value.strip()


def normalize_category(value: str) -> str:
    return remover_acentos(value).casefold().strip()


def load_truth_transactions(
    csv_path: str | Path,
    source_file: str | None = None,
) -> list[TruthTransaction]:
    path = Path(csv_path)
    occurrence_counter: dict[tuple[str, str, str, str, str], int] = {}
    transactions: list[TruthTransaction] = []

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            row_source_file = (row.get("source_file") or "").strip()
            if source_file and row_source_file != source_file:
                continue

            base_key = (
                row_source_file,
                (row.get("booking_date") or "").strip(),
                (row.get("description") or "").strip(),
                normalize_amount(row.get("amount") or ""),
                (row.get("direction") or "").strip(),
            )
            occurrence_counter[base_key] = occurrence_counter.get(base_key, 0) + 1

            transactions.append(
                TruthTransaction(
                    source_file=row_source_file,
                    bank_name=(row.get("bank_name") or "").strip(),
                    booking_date=(row.get("booking_date") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    amount=normalize_amount(row.get("amount") or ""),
                    direction=(row.get("direction") or "").strip(),
                    categoria_correta=(row.get("categoria_correta") or "").strip(),
                    extractor_status=(row.get("extractor_status") or "").strip(),
                    observacoes=(row.get("observacoes") or "").strip(),
                    occurrence=occurrence_counter[base_key],
                )
            )

    return transactions


def predict_transactions(
    source_files: list[str],
    model: str,
    categories: list[str],
    sem_ia: bool,
    usar_rag: bool,
    rag_csv: str,
    rag_db_path: str,
    rag_collection: str,
    rag_embedding_model: str,
    rag_top_k: int,
) -> list[PredictedTransaction]:
    transactions: list[PredictedTransaction] = []

    for source_file in source_files:
        payload = enriquecer_transacoes(
            pdf_path=source_file,
            model=model,
            categories=categories,
            sem_ia=sem_ia,
            usar_rag=usar_rag,
            rag_csv=rag_csv,
            rag_db_path=rag_db_path,
            rag_collection=rag_collection,
            rag_embedding_model=rag_embedding_model,
            rag_top_k=rag_top_k,
        )

        occurrence_counter: dict[tuple[str, str, str, str, str], int] = {}
        for transaction in payload["transactions"]:
            base_key = (
                str(transaction["source_file"]).strip(),
                str(transaction["booking_date"]).strip(),
                str(transaction["description"]).strip(),
                normalize_amount(str(transaction["amount"])),
                str(transaction["direction"]).strip(),
            )
            occurrence_counter[base_key] = occurrence_counter.get(base_key, 0) + 1

            transactions.append(
                PredictedTransaction(
                    source_file=str(transaction["source_file"]).strip(),
                    bank_name=str(transaction["bank_name"]).strip(),
                    booking_date=str(transaction["booking_date"]).strip(),
                    description=str(transaction["description"]).strip(),
                    amount=normalize_amount(str(transaction["amount"])),
                    direction=str(transaction["direction"]).strip(),
                    category=str(transaction["category"]).strip(),
                    classification_source=str(transaction["classification_source"]).strip(),
                    occurrence=occurrence_counter[base_key],
                )
            )

    return transactions


def build_report(
    truth_transactions: list[TruthTransaction],
    predicted_transactions: list[PredictedTransaction],
) -> str:
    truth_by_key = {transaction.key: transaction for transaction in truth_transactions}
    predicted_by_key = {transaction.key: transaction for transaction in predicted_transactions}

    matched_keys = [key for key in truth_by_key if key in predicted_by_key]
    expected_extractable = [
        transaction
        for transaction in truth_transactions
        if transaction.extractor_status != "missing_in_current_extractor"
    ]
    expected_extractable_keys = {transaction.key for transaction in expected_extractable}

    extraction_misses = [
        truth_by_key[key]
        for key in expected_extractable_keys
        if key not in predicted_by_key
    ]
    unexpected_predictions = [
        predicted
        for key, predicted in predicted_by_key.items()
        if key not in truth_by_key
    ]
    known_missing_rows = [
        transaction
        for transaction in truth_transactions
        if transaction.extractor_status == "missing_in_current_extractor"
    ]
    recovered_known_missing = [
        transaction
        for transaction in known_missing_rows
        if transaction.key in predicted_by_key
    ]

    category_hits = 0
    category_mismatches: list[tuple[TruthTransaction, PredictedTransaction]] = []
    for key in matched_keys:
        truth = truth_by_key[key]
        predicted = predicted_by_key[key]
        if normalize_category(truth.categoria_correta) == normalize_category(predicted.category):
            category_hits += 1
        else:
            category_mismatches.append((truth, predicted))

    total_truth = len(truth_transactions)
    total_predicted = len(predicted_transactions)
    total_extractable = len(expected_extractable)
    total_matched = len(matched_keys)
    extraction_recall = (
        (total_matched / total_extractable) * 100 if total_extractable else 0.0
    )
    classification_accuracy = (
        (category_hits / total_matched) * 100 if total_matched else 0.0
    )

    lines = [
        "=== Avaliacao do Modelo ===",
        f"Transacoes no CSV: {total_truth}",
        f"Transacoes previstas pelo pipeline: {total_predicted}",
        f"Transacoes esperadas para extracao: {total_extractable}",
        f"Transacoes emparelhadas: {total_matched}",
        f"Recall de extracao: {extraction_recall:.1f}%",
        f"Acerto de classificacao: {category_hits}/{total_matched} ({classification_accuracy:.1f}%)",
        f"Linhas marcadas como missing no CSV: {len(known_missing_rows)}",
        f"Linhas missing recuperadas pelo parser atual: {len(recovered_known_missing)}",
        f"Falhas de extracao: {len(extraction_misses)}",
        f"Transacoes inesperadas no output: {len(unexpected_predictions)}",
        f"Erros de classificacao: {len(category_mismatches)}",
    ]

    if extraction_misses:
        lines.append("")
        lines.append("Falhas de extracao:")
        for transaction in extraction_misses[:10]:
            lines.append(
                f"- {transaction.source_file} | {transaction.booking_date} | "
                f"{transaction.description} | esperado={transaction.categoria_correta}"
            )

    if unexpected_predictions:
        lines.append("")
        lines.append("Transacoes inesperadas no output:")
        for transaction in unexpected_predictions[:10]:
            lines.append(
                f"- {transaction.source_file} | {transaction.booking_date} | "
                f"{transaction.description} | previsto={transaction.category}"
            )

    if category_mismatches:
        lines.append("")
        lines.append("Erros de classificacao:")
        for truth, predicted in category_mismatches[:15]:
            lines.append(
                f"- {truth.source_file} | {truth.booking_date} | {truth.description} | "
                f"esperado={truth.categoria_correta} | previsto={predicted.category} | "
                f"fonte={predicted.classification_source}"
            )

    return "\n".join(lines)


def main() -> None:
    args = build_argument_parser().parse_args()
    truth_transactions = load_truth_transactions(
        csv_path=args.csv_path,
        source_file=args.source_file,
    )

    if not truth_transactions:
        raise ValueError("Nao foram encontradas transacoes no CSV para os filtros indicados.")

    source_files = sorted({transaction.source_file for transaction in truth_transactions})
    predicted_transactions = predict_transactions(
        source_files=source_files,
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

    report = build_report(truth_transactions, predicted_transactions)
    print(report)


if __name__ == "__main__":
    main()
