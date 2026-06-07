from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import unicodedata

import ollama

from extractor import extract_statement
from limpeza import limpar_descricao


DEFAULT_MODEL = "llama3"
DEFAULT_CATEGORIES = [
    "Supermercado",
    "Restauracao",
    "Transportes",
    "Saude",
    "Casa",
    "Tecnologia",
    "Subscricoes",
    "Cambio",
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
        description="Extrair movimentos do PDF e classifica-los com Ollama."
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


def remover_acentos(texto: str) -> str:
    normalized = unicodedata.normalize("NFKD", texto)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def classificar_transacao(
    description: str,
    cleaned_description: str,
    details: str | None,
    direction: str,
    model: str,
    categories: list[str],
) -> str:
    category_list = ", ".join(categories)
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classificas transacoes bancarias. "
                    "Responde apenas com o nome exato de uma categoria da lista fornecida. "
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
                    "Devolve apenas uma categoria."
                ),
            },
        ],
    )
    raw_category = response["message"]["content"]
    return normalizar_categoria(raw_category, categories)


def enriquecer_transacoes(
    pdf_path: str | Path,
    model: str,
    categories: list[str],
) -> dict[str, object]:
    extraction = extract_statement(pdf_path)
    cache: dict[str, str] = {}
    enriched_transactions: list[dict[str, str | None]] = []

    for transaction in extraction.transactions:
        transaction_record = transaction.to_record()
        cleaned_description = limpar_descricao(transaction.description)

        if cleaned_description not in cache:
            cache[cleaned_description] = classificar_transacao(
                description=transaction.description,
                cleaned_description=cleaned_description,
                details=transaction_record["details"],
                direction=transaction.direction,
                model=model,
                categories=categories,
            )

        transaction_record["clean_description"] = cleaned_description
        transaction_record["category"] = cache[cleaned_description]
        enriched_transactions.append(transaction_record)

    return {
        "metadata": extraction.metadata.to_record(),
        "classification": {
            "model": model,
            "categories": categories,
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
    )

    if args.format == "csv":
        write_csv(payload["transactions"], args.output)
        return

    write_json(payload, args.output)


if __name__ == "__main__":
    main()
