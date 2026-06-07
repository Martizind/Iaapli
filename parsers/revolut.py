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
        return self._is_legacy_statement(document) or self._is_consolidated_statement(document)

    def parse(self, document: PdfDocument) -> StatementExtraction:
        if self._is_legacy_statement(document):
            return self._parse_legacy_statement(document)
        return self._parse_consolidated_statement(document)

    def _is_legacy_statement(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return "Revolut Bank UAB" in text and "Account transactions from" in text

    def _is_consolidated_statement(self, document: PdfDocument) -> bool:
        text = document.full_text()
        return (
            "Revolut Bank UAB" in text
            and "Extrato personalizado" in text
            and "Extrato de operações" in text
        )

    def _parse_legacy_statement(self, document: PdfDocument) -> StatementExtraction:
        page = document.pages[0]
        rows = page.rows
        header_index = next(
            index
            for index, row in enumerate(rows)
            if "Date Description Money out Money in Balance" in row.text
        )
        header_row = rows[header_index]
        columns = self._resolve_legacy_columns(header_row)
        metadata = self._build_legacy_metadata(document, rows)

        transactions: list[StatementTransaction] = []
        current_transaction: StatementTransaction | None = None

        for row in rows[header_index + 1 :]:
            text = row.text
            if not text:
                continue
            if text.startswith("Report lost or stolen card"):
                break

            if self._is_legacy_transaction_row(text):
                if current_transaction is not None:
                    transactions.append(current_transaction)
                current_transaction = self._parse_legacy_transaction_row(
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

    def _parse_consolidated_statement(self, document: PdfDocument) -> StatementExtraction:
        metadata = self._build_consolidated_metadata(document)
        transactions: list[StatementTransaction] = []
        current_currency = "EUR"

        for page in document.pages:
            rows = page.rows
            account_currency = self._find_account_currency(rows)
            if account_currency is not None:
                current_currency = account_currency

            header_index = self._find_consolidated_header(rows)
            if header_index is None:
                continue

            columns = self._resolve_consolidated_columns(rows, header_index)
            current_transaction: StatementTransaction | None = None

            for row in rows[header_index + 1 :]:
                text = row.text
                if not text:
                    continue
                if text.startswith("Total"):
                    break

                if self._is_consolidated_transaction_row(row):
                    if current_transaction is not None:
                        transactions.append(current_transaction)
                    current_transaction = self._parse_consolidated_transaction_row(
                        row=row,
                        columns=columns,
                        source_file=document.filename,
                        currency=current_currency,
                    )
                    continue

                if current_transaction is not None and self._is_consolidated_detail_row(
                    row,
                    columns,
                ):
                    current_transaction.details.append(row.text)

            if current_transaction is not None:
                transactions.append(current_transaction)

        return StatementExtraction(metadata=metadata, transactions=transactions)

    def _build_legacy_metadata(
        self,
        document: PdfDocument,
        rows: list[PdfRow],
    ) -> StatementMetadata:
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

    def _build_consolidated_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        period_match = find_first(r"(\d{2}/\d{2}/\d{4}) - (\d{2}/\d{2}/\d{4})", text)
        iban_match = find_first(r"Número de conta \(IBAN\)\s+([A-Z0-9]+)", text)

        account_holder = None
        if document.pages:
            for row in document.pages[0].rows[:10]:
                if (
                    row.text
                    and row.text.upper() == row.text
                    and not any(char.isdigit() for char in row.text)
                    and len(row.text.split()) >= 2
                ):
                    account_holder = row.text
                    break

        statement_start = None
        statement_end = None
        if period_match:
            statement_start = datetime.strptime(period_match.group(1), "%d/%m/%Y").date()
            statement_end = datetime.strptime(period_match.group(2), "%d/%m/%Y").date()

        currencies = sorted(
            {
                match.group(1)
                for page in document.pages
                for row in page.rows
                for match in [re.match(r"Conta Pessoal \(([^)]+)\)", row.text)]
                if match
            }
        )
        metadata_currency = currencies[0] if len(currencies) == 1 else "MULTI"

        return StatementMetadata(
            bank_name=self.bank_name,
            source_file=document.filename,
            account_holder=account_holder,
            iban=iban_match.group(1) if iban_match else None,
            statement_start=statement_start,
            statement_end=statement_end,
            currency=metadata_currency,
        )

    def _resolve_legacy_columns(self, header_row: PdfRow) -> dict[str, float]:
        description_x = next(word.x0 for word in header_row.words if word.text == "Description")
        balance_x = next(word.x0 for word in header_row.words if word.text == "Balance")
        money_positions = sorted(word.x0 for word in header_row.words if word.text == "Money")

        return {
            "description": description_x,
            "money_out": money_positions[0],
            "money_in": money_positions[1],
            "balance": balance_x,
        }

    def _find_account_currency(self, rows: list[PdfRow]) -> str | None:
        for row in rows:
            match = re.match(r"Conta Pessoal \(([^)]+)\)", row.text)
            if match:
                return match.group(1).strip()
        return None

    def _find_consolidated_header(self, rows: list[PdfRow]) -> int | None:
        for index, row in enumerate(rows):
            if "Data Descrição Categoria Saldo Comissões" in row.text:
                return index
        return None

    def _resolve_consolidated_columns(
        self,
        rows: list[PdfRow],
        header_index: int,
    ) -> dict[str, float]:
        header_row = rows[header_index]
        amount_header_row = rows[header_index - 1] if header_index > 0 else header_row

        return {
            "description": next(word.x0 for word in header_row.words if word.text == "Descrição"),
            "category": next(word.x0 for word in header_row.words if word.text == "Categoria"),
            "amount": next(word.x0 for word in amount_header_row.words if word.text == "Dinheiro"),
            "balance": next(word.x0 for word in header_row.words if word.text == "Saldo"),
            "tax": next(word.x0 for word in amount_header_row.words if word.text == "Imposto"),
        }

    def _is_legacy_transaction_row(self, text: str) -> bool:
        return bool(re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s+", text))

    def _is_consolidated_transaction_row(self, row: PdfRow) -> bool:
        return bool(row.words) and bool(re.match(r"^\d{2}/\d{2}/\d{4}$", row.words[0].text))

    def _is_consolidated_detail_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
    ) -> bool:
        if not row.words or self._is_consolidated_transaction_row(row):
            return False

        first_word_x = row.words[0].x0
        return columns["description"] <= first_word_x < columns["amount"]

    def _parse_legacy_transaction_row(
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

    def _parse_consolidated_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        source_file: str,
        currency: str,
    ) -> StatementTransaction:
        date_text = row.text_between(x_max=columns["description"])
        description = row.text_between(
            x_min=columns["description"],
            x_max=columns["category"],
        )
        category = row.text_between(
            x_min=columns["category"],
            x_max=columns["amount"],
        )
        amount_text = row.text_between(
            x_min=columns["amount"],
            x_max=columns["balance"],
        )
        balance_text = row.text_between(
            x_min=columns["balance"],
            x_max=columns["tax"],
        )

        booking_date = datetime.strptime(date_text, "%d/%m/%Y").date()
        amount = parse_amount(amount_text)
        balance = parse_amount(balance_text)
        direction = "debit" if amount_text.strip().startswith("-") else "credit"

        if amount is None:
            raise ValueError(
                f"Could not parse Revolut consolidated amount from row: {row.text}"
            )

        details = [f"Categoria: {category}"] if category else []

        return StatementTransaction(
            bank_name=self.bank_name,
            source_file=source_file,
            page_number=row.page_number,
            booking_date=booking_date,
            description=description,
            amount=amount,
            direction=direction,
            balance=balance,
            currency=currency,
            details=details,
            raw_text=row.text,
        )
