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
)
from .layout import PdfDocument, PdfRow


class RevolutParser(StatementParser):
    bank_name = "Revolut"

    def can_parse(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return "Revolut Bank UAB" in text and "Account transactions from" in text

    def parse(self, document: PdfDocument) -> StatementExtraction:
        page = document.pages[0]
        rows = page.rows
        header_index = next(
            index
            for index, row in enumerate(rows)
            if "Date Description Money out Money in Balance" in row.text
        )
        header_row = rows[header_index]
        columns = self._resolve_columns(header_row)
        metadata = self._build_metadata(document, rows)

        transactions: list[StatementTransaction] = []
        current_transaction: StatementTransaction | None = None

        for row in rows[header_index + 1 :]:
            text = row.text
            if not text:
                continue
            if text.startswith("Report lost or stolen card"):
                break

            if self._is_transaction_row(text):
                if current_transaction is not None:
                    transactions.append(current_transaction)
                current_transaction = self._parse_transaction_row(
                    row=row,
                    columns=columns,
                    source_file=document.filename,
                )
                continue

            if current_transaction is not None and text.startswith(("Reference:", "From:", "To:")):
                current_transaction.details.append(text)

        if current_transaction is not None:
            transactions.append(current_transaction)

        return StatementExtraction(metadata=metadata, transactions=transactions)

    def _build_metadata(self, document: PdfDocument, rows: list[PdfRow]) -> StatementMetadata:
        text = document.full_text()
        period_match = find_first(
            r"Account transactions from ([A-Za-z]+ \d{1,2}, \d{4}) to ([A-Za-z]+ \d{1,2}, \d{4})",
            text,
        )
        iban_match = find_first(r"IBAN\s+([A-Z0-9]+)", text)

        account_holder = None
        for index, row in enumerate(rows):
            if row.text == "Revolut Bank UAB":
                for candidate in rows[index + 1 : index + 4]:
                    if candidate.text and candidate.text.upper() == candidate.text and "IBAN" not in candidate.text:
                        account_holder = candidate.text
                        break
                break

        statement_start = None
        statement_end = None
        if period_match:
            statement_start = datetime.strptime(period_match.group(1), "%B %d, %Y").date()
            statement_end = datetime.strptime(period_match.group(2), "%B %d, %Y").date()

        currency = None
        if rows and rows[0].text.endswith("Statement"):
            currency = rows[0].text.split()[0]

        return StatementMetadata(
            bank_name=self.bank_name,
            source_file=document.filename,
            account_holder=account_holder,
            iban=iban_match.group(1) if iban_match else None,
            statement_start=statement_start,
            statement_end=statement_end,
            currency=currency or "EUR",
        )

    def _resolve_columns(self, header_row: PdfRow) -> dict[str, float]:
        description_x = next(word.x0 for word in header_row.words if word.text == "Description")
        balance_x = next(word.x0 for word in header_row.words if word.text == "Balance")
        money_positions = sorted(word.x0 for word in header_row.words if word.text == "Money")

        return {
            "description": description_x,
            "money_out": money_positions[0],
            "money_in": money_positions[1],
            "balance": balance_x,
        }

    def _is_transaction_row(self, text: str) -> bool:
        return bool(re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s+", text))

    def _parse_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        source_file: str,
    ) -> StatementTransaction:
        date_text = row.text_between(x_max=columns["description"])
        description = row.text_between(
            x_min=columns["description"],
            x_max=columns["money_out"],
        )
        money_out = parse_amount(
            row.text_between(x_min=columns["money_out"], x_max=columns["money_in"])
        )
        money_in = parse_amount(
            row.text_between(x_min=columns["money_in"], x_max=columns["balance"])
        )
        balance = parse_amount(row.text_between(x_min=columns["balance"]))

        booking_date = datetime.strptime(date_text, "%b %d, %Y").date()
        amount = money_in if money_in is not None else money_out
        direction = "credit" if money_in is not None else "debit"

        if amount is None:
            raise ValueError(f"Could not parse Revolut amount from row: {row.text}")

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=source_file,
            page_number=row.page_number,
            booking_date=booking_date,
            description=description,
            amount=amount,
            direction=direction,
            balance=balance,
            currency="EUR",
            raw_text=row.text,
        )
