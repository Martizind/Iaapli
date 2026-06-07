from __future__ import annotations

from datetime import date
import re

from .base import (
    StatementExtraction,
    StatementMetadata,
    StatementParser,
    StatementTransaction,
    find_first,
    parse_amount,
    parse_day_month,
)
from .layout import PdfDocument


class QontoParser(StatementParser):
    bank_name = "Qonto"

    def can_parse(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return "QNTOFRP1XXX" in text or "Extratos da conta" in text

    def parse(self, document: PdfDocument) -> StatementExtraction:
        metadata = self._build_metadata(document)
        transactions: list[StatementTransaction] = []
        current_transaction: StatementTransaction | None = None
        in_transactions = False

        for page in document.pages:
            for row in page.rows_in_region(min_x=60, max_x=560, min_top=0, max_top=720):
                text = row.text
                if not text:
                    continue

                if text.startswith("Data de"):
                    in_transactions = True
                    continue

                if not in_transactions:
                    continue

                if text.startswith("liquid"):
                    continue
                if text.startswith("De ") and "/" in text:
                    continue
                if text.startswith("Todos os seus cart"):
                    current_transaction = None
                    continue
                if text.startswith("Qonto") or text.startswith("18 rue") or text.startswith("Prudentiel"):
                    continue

                if self._is_transaction_row(text):
                    if current_transaction is not None:
                        transactions.append(current_transaction)
                    current_transaction = self._parse_transaction_row(
                        text=text,
                        page_number=page.number,
                        source_file=document.filename,
                        statement_start=metadata.statement_start,
                    )
                    continue

                if current_transaction is not None:
                    current_transaction.details.append(text)

        if current_transaction is not None:
            transactions.append(current_transaction)

        return StatementExtraction(metadata=metadata, transactions=transactions)

    def _build_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        period_match = find_first(r"De (\d{2}/\d{2}/\d{4}) para (\d{2}/\d{2}/\d{4})", text)
        iban_match = find_first(r"IBAN:\s*([A-Z0-9 ]+)", text)

        statement_start = None
        statement_end = None
        if period_match:
            statement_start = date.fromisoformat(
                "-".join(reversed(period_match.group(1).split("/")))
            )
            statement_end = date.fromisoformat(
                "-".join(reversed(period_match.group(2).split("/")))
            )

        account_holder = None
        if document.pages:
            for row in document.pages[0].rows:
                if "Saldo em" in row.text:
                    candidate = row.text.split("Saldo em", maxsplit=1)[0].strip()
                    if candidate:
                        account_holder = candidate
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

    def _is_transaction_row(self, text: str) -> bool:
        return bool(re.match(r"^\d{2}/\d{2}\s+.+\s+[+-]\s+\d[\d.,]*\s+EUR$", text))

    def _parse_transaction_row(
        self,
        text: str,
        page_number: int,
        source_file: str,
        statement_start: date | None,
    ) -> StatementTransaction:
        match = re.match(
            r"^(?P<booking>\d{2}/\d{2})\s+(?P<description>.+?)\s+(?P<sign>[+-])\s+(?P<amount>\d[\d.,]*)\s+EUR$",
            text,
        )
        if not match:
            raise ValueError(f"Could not parse Qonto transaction row: {text}")

        year = statement_start.year if statement_start else date.today().year
        booking_date = parse_day_month(match.group("booking"), year)
        amount = parse_amount(match.group("amount"))

        if amount is None:
            raise ValueError(f"Could not parse Qonto amount from row: {text}")

        direction = "credit" if match.group("sign") == "+" else "debit"

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=source_file,
            page_number=page_number,
            booking_date=booking_date,
            description=match.group("description"),
            amount=amount,
            direction=direction,
            currency="EUR",
            raw_text=text,
        )
