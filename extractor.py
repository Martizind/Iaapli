from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from parsers import load_pdf_document, select_parser
from parsers.base import StatementExtraction


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Extract structured transactions from supported bank statement PDFs."
    )
    argument_parser.add_argument("pdf_path", help="Path to the PDF statement.")
    argument_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format. Defaults to json.",
    )
    argument_parser.add_argument(
        "--output",
        help="Optional path for the output file. If omitted, prints to stdout.",
    )
    return argument_parser


def extract_statement(pdf_path: str | Path) -> StatementExtraction:
    document = load_pdf_document(pdf_path)
    parser = select_parser(document)
    return parser.parse(document)


def extraction_to_payload(extraction: StatementExtraction) -> dict[str, object]:
    return extraction.to_payload()


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
    extraction = extract_statement(args.pdf_path)
    payload = extraction_to_payload(extraction)

    if args.format == "csv":
        write_csv(payload["transactions"], args.output)
        return

    write_json(payload, args.output)


if __name__ == "__main__":
    main()
