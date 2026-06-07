from __future__ import annotations

from abc import abstractmethod

from .base import (
    StatementExtraction,
    StatementMetadata,
    StatementParser,
    StatementTransaction,
)
from .layout import PdfDocument, PdfPage, PdfRow


class BaseTabularParser(StatementParser):
    def parse(self, document: PdfDocument) -> StatementExtraction:
        metadata = self.build_metadata(document)
        transactions: list[StatementTransaction] = []

        for page in document.pages:
            rows = self.page_rows(page)
            header_index = self.find_header_index(rows)
            if header_index is None:
                continue

            columns = self.resolve_columns(rows[header_index])
            for row in rows[header_index + 1 :]:
                if not row.text:
                    continue
                if self.should_stop_row(row):
                    break
                if not self.is_transaction_row(row):
                    continue
                transactions.append(
                    self.parse_transaction_row(
                        row=row,
                        columns=columns,
                        metadata=metadata,
                        document=document,
                    )
                )

        self.finalize_metadata(metadata, transactions)
        return StatementExtraction(metadata=metadata, transactions=transactions)

    def page_rows(self, page: PdfPage) -> list[PdfRow]:
        region = self.row_region()
        if region is None:
            return page.rows
        return page.rows_in_region(**region)

    def row_region(self) -> dict[str, float] | None:
        return None

    def find_header_index(self, rows: list[PdfRow]) -> int | None:
        for index, row in enumerate(rows):
            if self.is_header_row(row):
                return index
        return None

    def should_stop_row(self, row: PdfRow) -> bool:
        return False

    def finalize_metadata(
        self,
        metadata: StatementMetadata,
        transactions: list[StatementTransaction],
    ) -> None:
        if not transactions:
            return

        transaction_start = min(transaction.booking_date for transaction in transactions)
        transaction_end = max(transaction.booking_date for transaction in transactions)
        should_override = self.override_statement_bounds()

        if should_override or metadata.statement_start is None:
            metadata.statement_start = transaction_start
        if should_override or metadata.statement_end is None:
            metadata.statement_end = transaction_end

    def override_statement_bounds(self) -> bool:
        return False

    @abstractmethod
    def build_metadata(self, document: PdfDocument) -> StatementMetadata:
        raise NotImplementedError

    @abstractmethod
    def is_header_row(self, row: PdfRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def resolve_columns(self, header_row: PdfRow) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def is_transaction_row(self, row: PdfRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse_transaction_row(
        self,
        row: PdfRow,
        columns: dict[str, float],
        metadata: StatementMetadata,
        document: PdfDocument,
    ) -> StatementTransaction:
        raise NotImplementedError
