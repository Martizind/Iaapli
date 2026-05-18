from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
import unicodedata

from .base import (
    StatementExtraction,
    StatementMetadata,
    StatementParser,
    StatementTransaction,
    find_first,
    parse_amount,
)
from .layout import PdfDocument, PdfPage, PdfRow


DATE_TOKEN_PATTERN = re.compile(r"^\d{2}[./-]\d{2}(?:[./-]\d{2,4})?$")
AMOUNT_TOKEN_PATTERN = re.compile(r"^[+-]?\d[\d ]*[.,]\d{2}$")


@dataclass(frozen=True)
class GenericHeader:
    row_index: int
    style: str
    description_x: float
    movement_date_x: float | None = None
    value_date_x: float | None = None
    amount_x: float | None = None
    debit_x: float | None = None
    credit_x: float | None = None
    balance_x: float | None = None


def normalize_text(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value.replace("Ø", "O").replace("ª", "A"))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return " ".join(normalized.upper().split())


def is_date_token(value: str) -> bool:
    return bool(DATE_TOKEN_PATTERN.match(value))


def is_amount_token(value: str) -> bool:
    return bool(AMOUNT_TOKEN_PATTERN.match(value))


def parse_flexible_date(value: str, default_year: int | None = None) -> date:
    for fmt in (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%d.%m.%y",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    if default_year is not None:
        for fmt in ("%d-%m", "%d/%m", "%d.%m"):
            try:
                partial = datetime.strptime(value, fmt)
                return date(default_year, partial.month, partial.day)
            except ValueError:
                continue

    raise ValueError(f"Unsupported generic statement date: {value}")


class GenericStatementParser(StatementParser):
    bank_name = "Banco nao identificado"
    description_min_x = 100.0

    def can_parse(self, document: PdfDocument) -> bool:
        for page in document.pages:
            rows = self._page_rows(page)
            header = self._find_header(rows)
            if header is None:
                continue
            if self._find_first_transaction_row(rows, header.row_index + 1, header) is not None:
                return True
        return False

    def parse(self, document: PdfDocument) -> StatementExtraction:
        metadata = self._build_metadata(document)
        transactions: list[StatementTransaction] = []
        last_booking_date: date | None = None

        for page in document.pages:
            rows = self._page_rows(page)
            header = self._find_header(rows)
            if header is None:
                continue

            current_transaction: StatementTransaction | None = None
            first_transaction_row = self._find_first_transaction_row(
                rows,
                header.row_index + 1,
                header,
            )
            header = self._complete_header(header, first_transaction_row)

            for row in rows[header.row_index + 1 :]:
                if not row.text:
                    continue
                if self._should_stop_row(row):
                    break

                if self._is_transaction_row(row, header):
                    if current_transaction is not None:
                        transactions.append(current_transaction)
                    current_transaction = self._parse_transaction_row(
                        row=row,
                        header=header,
                        metadata=metadata,
                        source_file=document.filename,
                        previous_booking_date=last_booking_date,
                    )
                    last_booking_date = current_transaction.booking_date
                    continue

                if current_transaction is not None and self._is_detail_row(row, header):
                    current_transaction.details.append(row.text)

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

    def _page_rows(self, page: PdfPage) -> list[PdfRow]:
        return page.rows_in_region(
            min_x=20,
            max_x=max(560, page.width - 10),
            min_top=40,
            max_top=760,
        )

    def _build_metadata(self, document: PdfDocument) -> StatementMetadata:
        text = document.full_text()
        statement_start = None
        statement_end = None

        interval_patterns = (
            r"Per[ií]odo\s+[Dd]e (\d{2}[./-]\d{2}[./-]\d{4}) a (\d{2}[./-]\d{2}[./-]\d{4})",
            r"Intervalo de (\d{2}[./-]\d{2}[./-]\d{4}) a (\d{2}[./-]\d{2}[./-]\d{4})",
            r"\b[Dd]e (\d{2}[./-]\d{2}[./-]\d{4}) a (\d{2}[./-]\d{2}[./-]\d{4})",
        )
        for pattern in interval_patterns:
            match = find_first(pattern, text)
            if match:
                statement_start = parse_flexible_date(match.group(1))
                statement_end = parse_flexible_date(match.group(2))
                break

        if statement_start is None:
            previous_match = find_first(
                r"Data Extrato Anterior (\d{2}[./-]\d{2}[./-]\d{4})",
                text,
            )
            if previous_match:
                statement_start = parse_flexible_date(previous_match.group(1))

        if statement_end is None:
            current_match = find_first(
                r"Data Extrato Atual (\d{2}[./-]\d{2}[./-]\d{4})",
                text,
            )
            if current_match:
                statement_end = parse_flexible_date(current_match.group(1))

        iban_match = find_first(r"IBAN[:\s]+([A-Z0-9 ]+)", text)
        iban = None
        if iban_match:
            iban = " ".join(iban_match.group(1).split())

        bank_name = self.bank_name
        if document.pages:
            for row in document.pages[0].rows[:12]:
                left_text = row.text.split(",", maxsplit=1)[0].strip()
                normalized = normalize_text(left_text)
                if (
                    left_text
                    and len(left_text.split()) <= 6
                    and any(token in normalized for token in ("BANCO", "BANK"))
                ):
                    bank_name = left_text
                    break

        if bank_name == self.bank_name:
            normalized_text = normalize_text(text)
            if "BANCOBPI.PT" in normalized_text or "BPI DIRETO" in normalized_text:
                bank_name = "Banco BPI"

        return StatementMetadata(
            bank_name=bank_name,
            source_file=document.filename,
            iban=iban,
            statement_start=statement_start,
            statement_end=statement_end,
            currency="EUR",
        )

    def _find_header(self, rows: list[PdfRow]) -> GenericHeader | None:
        for index, row in enumerate(rows):
            normalized_row = normalize_text(row.text)
            normalized_prev = normalize_text(rows[index - 1].text) if index > 0 else ""
            normalized_words = [normalize_text(word.text) for word in row.words]

            has_description = any(
                token.startswith("DESCR") or token == "DESCRIPTION"
                for token in normalized_words
            )
            has_balance = "SALDO" in normalized_row or "BALANCE" in normalized_row
            has_balance = has_balance or "SALDO" in normalized_prev or "BALANCE" in normalized_prev
            has_amount = (
                "MONTANTE" in normalized_row
                or "AMOUNT" in normalized_row
                or "VALOR" in normalized_row
            )
            has_debit = any(
                token == "DEBIT" or "BITO" in token
                for token in normalized_words
            )
            has_credit = any(
                token == "CREDIT"
                or token.startswith("CR")
                or "REDITO" in token
                or "DITO" in token and "BITO" not in token
                for token in normalized_words
            )
            has_date = "DATA" in normalized_row or "DATE" in normalized_row

            if has_date and has_description and has_debit and has_credit and has_balance:
                description_x = self._description_x(row)
                if description_x is None:
                    continue
                date_indexes = [
                    position
                    for position, token in enumerate(normalized_words)
                    if token == "DATA" or token == "DATE"
                ]
                balance_index = next(
                    (
                        position
                        for position, token in enumerate(normalized_words)
                        if token.startswith("SALDO") or token == "BALANCE"
                    ),
                    None,
                )
                debit_index = next(
                    (
                        position
                        for position, token in enumerate(normalized_words)
                        if token == "DEBIT" or "BITO" in token
                    ),
                    None,
                )
                if balance_index is None or debit_index is None:
                    continue
                credit_index = next(
                    (
                        position
                        for position, token in enumerate(normalized_words)
                        if position > debit_index
                        and position < balance_index
                        and (
                            token == "CREDIT"
                            or token.startswith("CR")
                            or "REDITO" in token
                            or ("DITO" in token and "BITO" not in token)
                        )
                    ),
                    balance_index - 1,
                )
                return GenericHeader(
                    row_index=index,
                    style="dual",
                    description_x=description_x,
                    movement_date_x=row.words[date_indexes[0]].x0 if date_indexes else None,
                    value_date_x=row.words[date_indexes[1]].x0 if len(date_indexes) > 1 else None,
                    debit_x=row.words[debit_index].x0 - 5,
                    credit_x=row.words[credit_index].x0 - 5,
                    balance_x=row.words[balance_index].x0 - 5,
                )

            if has_date and has_description and has_amount and has_balance:
                description_x = self._description_x(row)
                if description_x is None:
                    continue
                date_indexes = [
                    position
                    for position, token in enumerate(normalized_words)
                    if token == "DATA" or token == "DATE"
                ]
                amount_x = self._word_x(
                    row,
                    lambda token: (
                        token.startswith("MONTANTE")
                        or token == "AMOUNT"
                        or token.startswith("VALOR")
                    ),
                )
                balance_x = self._word_x(
                    row,
                    lambda token: token.startswith("SALDO") or token == "BALANCE",
                )
                return GenericHeader(
                    row_index=index,
                    style="single",
                    description_x=description_x,
                    movement_date_x=row.words[date_indexes[0]].x0 if date_indexes else None,
                    value_date_x=row.words[date_indexes[1]].x0 if len(date_indexes) > 1 else None,
                    amount_x=amount_x,
                    balance_x=balance_x,
                )

        return None

    def _find_first_transaction_row(
        self,
        rows: list[PdfRow],
        start_index: int,
        header: GenericHeader | None = None,
    ) -> PdfRow | None:
        for row in rows[start_index:]:
            if self._should_stop_row(row):
                break
            if self._is_transaction_row(row, header):
                return row
        return None

    def _complete_header(
        self,
        header: GenericHeader,
        first_transaction_row: PdfRow | None,
    ) -> GenericHeader:
        if first_transaction_row is None:
            return header

        description_x = header.description_x
        if len(first_transaction_row.words) >= 3:
            description_x = min(description_x, first_transaction_row.words[2].x0)

        if header.style == "dual":
            amount_words = [word for word in first_transaction_row.words if is_amount_token(word.text)]
            if len(amount_words) < 2:
                return header
            balance_x = header.balance_x or amount_words[-1].x0 - 5
            credit_x = header.credit_x or amount_words[-2].x0 - 5
            debit_x = header.debit_x
            if debit_x is None and len(amount_words) >= 3:
                debit_x = amount_words[-3].x0 - 5
            if debit_x is None:
                debit_x = (credit_x + header.description_x) / 2

            return GenericHeader(
                row_index=header.row_index,
                style=header.style,
                description_x=description_x,
                movement_date_x=header.movement_date_x,
                value_date_x=header.value_date_x,
                debit_x=debit_x,
                credit_x=credit_x,
                balance_x=balance_x,
            )

        amount_words = [word for word in first_transaction_row.words if is_amount_token(word.text)]
        if len(amount_words) < 2:
            return header

        amount_x = header.amount_x or amount_words[-2].x0 - 5
        balance_x = header.balance_x or amount_words[-1].x0 - 5
        return GenericHeader(
            row_index=header.row_index,
            style=header.style,
            description_x=description_x,
            movement_date_x=header.movement_date_x,
            value_date_x=header.value_date_x,
            amount_x=amount_x,
            balance_x=balance_x,
        )

    def _description_x(self, row: PdfRow) -> float | None:
        description_x = self._word_x(
            row,
            lambda token: token.startswith("DESCR") or token == "DESCRIPTION",
        )
        if description_x is not None:
            return description_x
        return self.description_min_x

    def _word_x(
        self,
        row: PdfRow,
        predicate,
    ) -> float | None:
        for word in row.words:
            if predicate(normalize_text(word.text)):
                return word.x0
        return None

    def _date_words_before_description(
        self,
        row: PdfRow,
        header: GenericHeader,
    ) -> list:
        return [
            word
            for word in row.words
            if word.x0 < header.description_x and is_date_token(word.text)
        ]

    def _is_transaction_row(
        self,
        row: PdfRow,
        header: GenericHeader | None = None,
    ) -> bool:
        if len(row.words) < 3:
            return False

        if header is None:
            return is_date_token(row.words[0].text) and is_date_token(row.words[1].text)

        date_words = self._date_words_before_description(row, header)
        if not date_words:
            return False

        amount_boundary = header.debit_x or header.amount_x or header.balance_x or 380
        amount_words = [
            word
            for word in row.words
            if word.x0 >= amount_boundary and is_amount_token(word.text)
        ]
        required_amounts = 2 if header.style == "dual" else 1
        return len(amount_words) >= required_amounts

    def _is_detail_row(self, row: PdfRow, header: GenericHeader) -> bool:
        if self._is_transaction_row(row, header) or not row.words:
            return False

        first_word_x = row.words[0].x0
        limit_x = header.debit_x or header.amount_x or header.balance_x or 560
        return header.description_x <= first_word_x < limit_x

    def _should_stop_row(self, row: PdfRow) -> bool:
        normalized = normalize_text(row.text)
        return normalized.startswith(
            (
                "TOTAL",
                "SALDO FINAL",
                "SALDO ACTUAL",
                "SALDO CONTABIL",
                "SALDO DISPONIVEL",
                "DETALHE DO PATRIM",
                "FUNDO DE GARANTIA",
                "OUTRAS INFORMACOES",
                "INFORMACOES IMPORTANTES",
            )
        )

    def _extract_trailing_amount_groups(
        self,
        row: PdfRow,
        expected_groups: int,
    ) -> list[list]:
        groups: list[list] = []
        index = len(row.words) - 1

        while index >= 0 and len(groups) < expected_groups:
            word = row.words[index]
            if not is_amount_token(word.text):
                index -= 1
                continue

            group = [word]
            index -= 1

            while index >= 0:
                previous_word = row.words[index]
                candidate = [previous_word, *group]
                candidate_text = normalize_text(" ".join(item.text for item in candidate))
                gap = group[0].x0 - previous_word.x1

                if gap > 4 or parse_amount(candidate_text) is None:
                    break

                group = candidate
                index -= 1

            groups.append(group)

        groups.reverse()
        return groups

    def _resolve_transaction_dates(
        self,
        row: PdfRow,
        header: GenericHeader,
        default_year: int,
        previous_booking_date: date | None,
    ) -> tuple[date, date]:
        date_words = self._date_words_before_description(row, header)
        if not date_words:
            raise ValueError(f"Could not parse generic transaction dates: {row.text}")

        booking_token: str | None = None
        value_token: str | None = None

        if header.value_date_x is not None:
            for word in date_words:
                if word.x0 >= header.value_date_x - 1:
                    if value_token is None:
                        value_token = word.text
                    continue
                if booking_token is None:
                    booking_token = word.text
        else:
            booking_token = date_words[0].text
            if len(date_words) > 1:
                value_token = date_words[1].text

        if booking_token is None:
            if previous_booking_date is not None:
                booking_date = previous_booking_date
            elif value_token is not None:
                booking_date = parse_flexible_date(value_token, default_year=default_year)
            else:
                raise ValueError(f"Could not infer generic booking date: {row.text}")
        else:
            booking_date = parse_flexible_date(booking_token, default_year=default_year)

        if value_token is None:
            value_date = booking_date
        else:
            value_date = parse_flexible_date(value_token, default_year=default_year)

        return booking_date, value_date

    def _parse_transaction_row(
        self,
        row: PdfRow,
        header: GenericHeader,
        metadata: StatementMetadata,
        source_file: str,
        previous_booking_date: date | None = None,
    ) -> StatementTransaction:
        default_year = None
        if metadata.statement_end is not None:
            default_year = metadata.statement_end.year
        elif metadata.statement_start is not None:
            default_year = metadata.statement_start.year
        else:
            default_year = datetime.today().year

        booking_date, value_date = self._resolve_transaction_dates(
            row=row,
            header=header,
            default_year=default_year,
            previous_booking_date=previous_booking_date,
        )

        if header.style == "dual":
            debit_x = header.debit_x or 380
            credit_x = header.credit_x or debit_x + 60
            balance_x = header.balance_x or credit_x + 60
            description = row.text_between(
                x_min=header.description_x,
                x_max=debit_x,
            )
            debit = parse_amount(row.text_between(x_min=debit_x, x_max=credit_x))
            credit = parse_amount(row.text_between(x_min=credit_x, x_max=balance_x))
            balance = parse_amount(row.text_between(x_min=balance_x))
            amount = credit if credit is not None else debit
            direction = "credit" if credit is not None else "debit"
        else:
            amount_x = header.amount_x or 380
            balance_x = header.balance_x or amount_x + 100
            amount_groups = self._extract_trailing_amount_groups(row, expected_groups=2)

            if len(amount_groups) >= 2:
                amount_words = amount_groups[-2]
                balance_words = amount_groups[-1]
                description = row.text_between(
                    x_min=header.description_x,
                    x_max=amount_words[0].x0,
                )
                amount_text = normalize_text(" ".join(word.text for word in amount_words))
                balance_text = normalize_text(" ".join(word.text for word in balance_words))
                amount = parse_amount(amount_text)
                balance = parse_amount(balance_text)
            else:
                description = row.text_between(
                    x_min=header.description_x,
                    x_max=amount_x,
                )
                amount_text = row.text_between(x_min=amount_x, x_max=balance_x)
                amount = parse_amount(amount_text)
                balance = parse_amount(row.text_between(x_min=balance_x))

            direction = "debit" if amount_text.strip().startswith("-") else "credit"

        if amount is None:
            raise ValueError(f"Could not parse generic transaction row: {row.text}")

        return StatementTransaction(
            bank_name=metadata.bank_name,
            source_file=source_file,
            page_number=row.page_number,
            booking_date=booking_date,
            value_date=value_date,
            description=description,
            amount=amount,
            direction=direction,
            balance=balance,
            currency=metadata.currency or "EUR",
            raw_text=row.text,
        )
