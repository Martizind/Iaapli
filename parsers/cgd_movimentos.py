from __future__ import annotations

import re

from .base import (
    StatementMetadata,
    StatementTransaction,
    find_first,
    parse_amount,
    parse_iso_date,
)
from .base_tabular import BaseTabularParser
from .layout import PdfDocument, PdfRow


class CgdMovimentosParser(BaseTabularParser):
    bank_name = "Caixa Geral de Depositos"
    transaction_pattern = re.compile(
        r"^(?P<booking>\d{2}-\d{2}-\d{4})\s+"
        r"(?P<value>\d{2}-\d{2}-\d{4})\s+"
        r"(?P<description>.+?)\s+"
        r"(?P<amount>-?\d[\d.,]*)\s+"
        r"(?P<balance>-?\d[\d.,]*)$"
    )

    def can_parse(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return (
            "Consultar saldos e movimentos" in text
            and "Data mov. Data-valor" in text
            and "Caixa Geral de Dep" in text
        )

    def row_region(self) -> dict[str, float]:
        return {
            "min_x": 55,
            "max_x": 545,
            "min_top": 0,
            "max_top": 770,
        }

    def build_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        period_match = find_first(
            r"Intervalo de (\d{2}-\d{2}-\d{4}) a (\d{2}-\d{2}-\d{4})",
            text,
        )

        account_holder = None
        if document.pages:
            for row in document.pages[0].rows:
                if row.text.startswith("Cliente "):
                    account_holder = row.text.replace("Cliente ", "", 1).strip()
                    break

        statement_start = None
        statement_end = None
        if period_match:
            statement_start = parse_iso_date(period_match.group(1), "%d-%m-%Y")
            statement_end = parse_iso_date(period_match.group(2), "%d-%m-%Y")

        return StatementMetadata(
            bank_name=self.bank_name,
            source_file=document.filename,
            account_holder=account_holder,
            statement_start=statement_start,
            statement_end=statement_end,
            currency="EUR",
        )

    def is_header_row(self, row: PdfRow) -> bool:
        return (
            "Data mov." in row.text
            and "Data-valor" in row.text
            and "Montante" in row.text
        )

    def resolve_columns(self, header_row: PdfRow) -> dict[str, float]:
        return {}

    def should_stop_row(self, row: PdfRow) -> bool:
        return row.text.startswith("Caixa Geral de Dep")

    def is_transaction_row(self, row: PdfRow) -> bool:
        return bool(self.transaction_pattern.match(row.text))

    def parse_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        metadata: StatementMetadata,
        document: PdfDocument,
    ) -> StatementTransaction:
        match = self.transaction_pattern.match(row.text)
        if match is None:
            raise ValueError(f"Could not parse CGD transaction row: {row.text}")

        booking_date = parse_iso_date(match.group("booking"), "%d-%m-%Y")
        value_date = parse_iso_date(match.group("value"), "%d-%m-%Y")
        amount_text = match.group("amount")
        amount = parse_amount(amount_text)
        balance = parse_amount(match.group("balance"))

        if amount is None:
            raise ValueError(f"Could not parse CGD amount from row: {row.text}")

        direction = "debit" if amount_text.startswith("-") else "credit"

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=document.filename,
            page_number=row.page_number,
            booking_date=booking_date,
            value_date=value_date,
            description=match.group("description"),
            amount=amount,
            direction=direction,
            balance=balance,
            currency="EUR",
            raw_text=row.text,
        )
