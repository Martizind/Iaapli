from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re

from .layout import PdfDocument


def parse_amount(value: str) -> Decimal | None:
    cleaned = (
        value.replace("EUR", "")
        .replace("€", "")
        .replace("+", "")
        .replace("-", "")
        .replace(" ", "")
        .strip()
    )

    if not cleaned:
        return None

    cleaned = cleaned.replace(",", ".")

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_iso_date(value: str, fmt: str) -> date:
    return datetime.strptime(value, fmt).date()


def parse_day_month(value: str, year: int) -> date:
    return datetime.strptime(f"{value}/{year}", "%d/%m/%Y").date()


def parse_month_day_dot(value: str, year: int) -> date:
    month, day = value.split(".")
    return date(year, int(month), int(day))


def find_first(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.MULTILINE)


@dataclass
class StatementMetadata:
    bank_name: str
    source_file: str
    account_holder: str | None = None
    iban: str | None = None
    statement_start: date | None = None
    statement_end: date | None = None
    currency: str | None = "EUR"

    def to_record(self) -> dict[str, str | None]:
        return {
            "bank_name": self.bank_name,
            "source_file": self.source_file,
            "account_holder": self.account_holder,
            "iban": self.iban,
            "statement_start": self.statement_start.isoformat() if self.statement_start else None,
            "statement_end": self.statement_end.isoformat() if self.statement_end else None,
            "currency": self.currency,
        }


@dataclass
class StatementTransaction:
    bank_name: str
    source_file: str
    page_number: int
    booking_date: date
    description: str
    amount: Decimal
    direction: str
    currency: str = "EUR"
    value_date: date | None = None
    balance: Decimal | None = None
    details: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_record(self) -> dict[str, str | None]:
        return {
            "bank_name": self.bank_name,
            "source_file": self.source_file,
            "page_number": str(self.page_number),
            "booking_date": self.booking_date.isoformat(),
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "description": self.description,
            "amount": f"{self.amount:.2f}",
            "direction": self.direction,
            "balance": f"{self.balance:.2f}" if self.balance is not None else None,
            "currency": self.currency,
            "details": " | ".join(self.details) if self.details else None,
            "raw_text": self.raw_text or self.description,
        }


@dataclass
class StatementExtraction:
    metadata: StatementMetadata
    transactions: list[StatementTransaction]

    def to_payload(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_record(),
            "transactions": [transaction.to_record() for transaction in self.transactions],
        }


class StatementParser(ABC):
    bank_name: str

    @abstractmethod
    def can_parse(self, document: PdfDocument) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, document: PdfDocument) -> StatementExtraction:
        raise NotImplementedError
