from __future__ import annotations

from datetime import datetime
import re

from .base import (
    StatementExtraction,
    StatementMetadata,
    StatementParser,
    StatementTransaction,
    find_first,
    parse_amount,
    parse_iso_date,
)
from .layout import PdfDocument, PdfRow


class NovoBancoParser(StatementParser):
    bank_name = "Novo Banco"
    transaction_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{2}\s+\d{2}\.\d{2}\.\d{2}\s+")
    description_min_x = 100.0

    def can_parse(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return (
            "EXTRATO INTEGRADO" in text
            and "MOVIMENTOS DE CONTA" in text
            and "BESCPTPL" in text
        )

    def parse(self, document: PdfDocument) -> StatementExtraction:
        metadata = self._build_metadata(document)
        transactions: list[StatementTransaction] = []

        for page in document.pages:
            rows = page.rows_in_region(min_x=20, max_x=550, min_top=45, max_top=390)
            header_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if "Data Valor" in row.text
                    and "Descritivo" in row.text
                    and "Saldo" in row.text
                ),
                None,
            )
            if header_index is None:
                continue

            columns = self._resolve_columns(rows[header_index])
            current_transaction: StatementTransaction | None = None

            for row in rows[header_index + 1 :]:
                text = row.text
                if not text:
                    continue
                if text.startswith(("TOTAL", "SALDO CONTABIL", "DETALHE DO PATRIM")):
                    break

                if self._is_transaction_row(row):
                    if current_transaction is not None:
                        transactions.append(current_transaction)
                    current_transaction = self._parse_transaction_row(
                        row=row,
                        columns=columns,
                        source_file=document.filename,
                    )
                    continue

                if current_transaction is not None and self._is_detail_row(row, columns):
                    current_transaction.details.append(text)

            if current_transaction is not None:
                transactions.append(current_transaction)

        if transactions and metadata.statement_start is None:
            metadata.statement_start = min(
                transaction.booking_date for transaction in transactions
            )
        if transactions and metadata.statement_end is None:
            metadata.statement_end = max(
                transaction.booking_date for transaction in transactions
            )

        return StatementExtraction(metadata=metadata, transactions=transactions)

    def _build_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        current_match = find_first(r"Data Extrato Atual (\d{2}\.\d{2}\.\d{4})", text)
        previous_match = find_first(r"Data Extrato Anterior (\d{2}\.\d{2}\.\d{4})", text)
        iban_match = find_first(r"IBAN\s+([A-Z0-9 ]+)", text)

        statement_start = None
        statement_end = None
        if previous_match:
            statement_start = parse_iso_date(previous_match.group(1), "%d.%m.%Y")
        if current_match:
            statement_end = parse_iso_date(current_match.group(1), "%d.%m.%Y")

        account_holder = None
        if document.pages:
            first_page_rows = document.pages[0].rows_in_region(
                min_x=40,
                max_x=550,
                min_top=140,
                max_top=190,
            )
            for row in first_page_rows:
                right_text = row.text_between(x_min=300)
                if (
                    right_text
                    and right_text.upper() == right_text
                    and not right_text.startswith("(")
                    and not right_text.startswith(("CAM ", "FUNCHAL", "4910-"))
                    and "BESCPTPL" not in right_text
                    and "Data Extrato" not in row.text
                ):
                    account_holder = right_text
                    break

        iban = None
        if iban_match:
            iban = " ".join(iban_match.group(1).split())

        return StatementMetadata(
            bank_name=self.bank_name,
            source_file=document.filename,
            account_holder=account_holder,
            iban=iban,
            statement_start=statement_start,
            statement_end=statement_end,
            currency="EUR",
        )

    def _resolve_columns(self, header_row: PdfRow) -> dict[str, float]:
        balance_index = next(
            index for index, word in enumerate(header_row.words) if word.text == "Saldo"
        )
        debit_index = next(
            index
            for index, word in enumerate(header_row.words)
            if "bito" in word.text.casefold()
        )
        debit_x = header_row.words[debit_index].x0
        credit_x = header_row.words[balance_index - 1].x0
        balance_x = header_row.words[balance_index].x0

        return {
            "debit": debit_x - 5,
            "credit": credit_x - 5,
            "balance": balance_x - 5,
            "description": self.description_min_x,
        }

    def _is_transaction_row(self, row: PdfRow) -> bool:
        return bool(self.transaction_pattern.match(row.text))

    def _is_detail_row(self, row: PdfRow, columns: dict[str, float]) -> bool:
        if self._is_transaction_row(row):
            return False
        first_word_x = row.words[0].x0 if row.words else 0
        return columns["description"] <= first_word_x < columns["debit"]

    def _parse_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        source_file: str,
    ) -> StatementTransaction:
        if len(row.words) < 4:
            raise ValueError(f"Could not parse Novo Banco row: {row.text}")

        booking_text = row.words[0].text
        value_text = row.words[1].text
        description_parts = [
            word.text
            for word in row.words[2:]
            if word.x0 < columns["debit"]
        ]
        description = " ".join(description_parts).strip()
        debit = parse_amount(
            row.text_between(x_min=columns["debit"], x_max=columns["credit"])
        )
        credit = parse_amount(
            row.text_between(x_min=columns["credit"], x_max=columns["balance"])
        )
        balance = parse_amount(row.text_between(x_min=columns["balance"]))

        booking_date = datetime.strptime(booking_text, "%d.%m.%y").date()
        value_date = datetime.strptime(value_text, "%d.%m.%y").date()
        amount = credit if credit is not None else debit
        direction = "credit" if credit is not None else "debit"

        if amount is None:
            raise ValueError(f"Could not parse Novo Banco amount from row: {row.text}")

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=source_file,
            page_number=row.page_number,
            booking_date=booking_date,
            value_date=value_date,
            description=description,
            amount=amount,
            direction=direction,
            balance=balance,
            currency="EUR",
            raw_text=row.text,
        )
