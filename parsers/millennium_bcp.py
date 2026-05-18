from __future__ import annotations

from datetime import datetime
import re

from .base import (
    StatementMetadata,
    StatementTransaction,
    find_first,
    parse_amount,
    parse_month_day_dot,
)
from .base_tabular import BaseTabularParser
from .layout import PdfDocument, PdfRow


class MillenniumBcpParser(BaseTabularParser):
    bank_name = "Millennium BCP"
    transaction_pattern = re.compile(r"^\d{2}\.\d{2}\s+\d{2}\.\d{2}\s+")

    def can_parse(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return "EXTRATO COMBINADO" in text and "BCOMPTPL" in text

    def row_region(self) -> dict[str, float]:
        return {
            "min_x": 50,
            "max_x": 560,
            "min_top": 40,
            "max_top": 760,
        }

    def build_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        statement_date_match = find_first(r"\b(\d{2}/\d{2}/\d{2})\s+CONTA:", text)
        iban_match = find_first(r"IBAN:\s*([A-Z0-9 ]+)", text)

        statement_end = None
        if statement_date_match:
            statement_end = datetime.strptime(statement_date_match.group(1), "%y/%m/%d").date()

        account_holder = None
        if document.pages:
            for row in document.pages[0].rows_in_region(min_x=40, max_x=360, min_top=220, max_top=340):
                if "MEMORIA" in row.text and "LDA" in row.text:
                    account_holder = row.text.replace("#S#", "").replace("#E#", "").strip()
                    break

        iban = None
        if iban_match:
            iban = " ".join(iban_match.group(1).split())

        return StatementMetadata(
            bank_name=self.bank_name,
            source_file=document.filename,
            account_holder=account_holder,
            iban=iban,
            statement_end=statement_end,
            currency="EUR",
        )

    def is_header_row(self, row: PdfRow) -> bool:
        return "LANC." in row.text and "DESCRITIVO" in row.text and "SALDO" in row.text

    def resolve_columns(self, header_row: PdfRow) -> dict[str, float]:
        value_date_x = next(word.x0 for word in header_row.words if word.text == "VALOR")
        description_x = next(word.x0 for word in header_row.words if word.text == "DESCRITIVO")
        debit_x = next(word.x0 for word in header_row.words if word.text == "DEBITO")
        credit_x = next(word.x0 for word in header_row.words if word.text == "CREDITO")
        balance_x = next(word.x0 for word in header_row.words if word.text == "SALDO")
        value_date_end = (value_date_x + description_x) / 2
        debit_start = debit_x - 15
        credit_start = credit_x - 15
        balance_start = balance_x - 20

        return {
            "booking_date_end": value_date_x,
            "value_date_start": value_date_x,
            "value_date_end": value_date_end,
            "description": value_date_end,
            "description_end": debit_start,
            "debit": debit_start,
            "credit": credit_start,
            "balance": balance_start,
        }

    def should_stop_row(self, row: PdfRow) -> bool:
        return row.text.startswith("SALDO FINAL")

    def is_transaction_row(self, row: PdfRow) -> bool:
        return bool(self.transaction_pattern.match(row.text))

    def parse_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        metadata: StatementMetadata,
        document: PdfDocument,
    ) -> StatementTransaction:
        booking_text = row.text_between(x_max=columns["booking_date_end"])
        value_text = row.text_between(
            x_min=columns["value_date_start"],
            x_max=columns["value_date_end"],
        )
        description = row.text_between(
            x_min=columns["description"],
            x_max=columns["description_end"],
        )
        debit = parse_amount(row.text_between(x_min=columns["debit"], x_max=columns["credit"]))
        credit = parse_amount(row.text_between(x_min=columns["credit"], x_max=columns["balance"]))
        balance = parse_amount(row.text_between(x_min=columns["balance"]))

        statement_year = metadata.statement_end.year if metadata.statement_end else None
        if statement_year is None:
            raise ValueError("Millennium BCP parser needs a statement year.")

        booking_date = parse_month_day_dot(booking_text, statement_year)
        value_date = parse_month_day_dot(value_text, statement_year)
        amount = credit if credit is not None else debit
        direction = "credit" if credit is not None else "debit"

        if amount is None:
            raise ValueError(f"Could not parse Millennium BCP amount from row: {row.text}")

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=document.filename,
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

    def override_statement_bounds(self) -> bool:
        return True
