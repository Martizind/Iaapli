from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class PdfWord:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    page_number: int


@dataclass
class PdfRow:
    page_number: int
    page_width: float
    top: float
    bottom: float
    words: list[PdfWord]

    def __post_init__(self) -> None:
        self.words.sort(key=lambda word: (word.x0, word.top))

    @property
    def text(self) -> str:
        return normalize_spaces(" ".join(word.text for word in self.words))

    def text_between(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> str:
        selected = []
        for word in self.words:
            if x_min is not None and word.x0 < x_min:
                continue
            if x_max is not None and word.x0 >= x_max:
                continue
            selected.append(word.text)
        return normalize_spaces(" ".join(selected))


@dataclass
class PdfPage:
    number: int
    width: float
    height: float
    words: list[PdfWord]
    rows: list[PdfRow]

    def rows_in_region(
        self,
        min_x: float | None = None,
        max_x: float | None = None,
        min_top: float | None = None,
        max_top: float | None = None,
    ) -> list[PdfRow]:
        region_rows: list[PdfRow] = []
        for row in self.rows:
            if min_top is not None and row.bottom < min_top:
                continue
            if max_top is not None and row.top > max_top:
                continue

            selected_words = []
            for word in row.words:
                if min_x is not None and word.x0 < min_x:
                    continue
                if max_x is not None and word.x1 > max_x:
                    continue
                selected_words.append(word)

            if not selected_words:
                continue

            region_rows.append(
                PdfRow(
                    page_number=row.page_number,
                    page_width=row.page_width,
                    top=row.top,
                    bottom=row.bottom,
                    words=selected_words,
                )
            )
        return region_rows


@dataclass
class PdfDocument:
    path: Path
    pages: list[PdfPage]

    @property
    def filename(self) -> str:
        return self.path.name

    def full_text(self) -> str:
        lines = []
        for page in self.pages:
            lines.extend(row.text for row in page.rows if row.text)
        return "\n".join(lines)


def _build_rows(
    words: list[PdfWord],
    page_number: int,
    page_width: float,
    row_tolerance: float = 4.0,
) -> list[PdfRow]:
    sorted_words = sorted(words, key=lambda word: (word.top, word.x0))
    rows: list[list[PdfWord]] = []
    current_row: list[PdfWord] = []
    current_top: float | None = None

    for word in sorted_words:
        if not current_row:
            current_row = [word]
            current_top = word.top
            continue

        if current_top is not None and abs(word.top - current_top) <= row_tolerance:
            current_row.append(word)
            current_top = sum(item.top for item in current_row) / len(current_row)
            continue

        rows.append(current_row)
        current_row = [word]
        current_top = word.top

    if current_row:
        rows.append(current_row)

    return [
        PdfRow(
            page_number=page_number,
            page_width=page_width,
            top=min(word.top for word in row_words),
            bottom=max(word.bottom for word in row_words),
            words=row_words,
        )
        for row_words in rows
    ]


def load_pdf_document(
    path: str | Path,
    x_tolerance: float = 2.0,
    y_tolerance: float = 2.0,
) -> PdfDocument:
    pdf_path = Path(path)
    pages: list[PdfPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False,
                use_text_flow=False,
            )

            words = [
                PdfWord(
                    text=normalize_spaces(raw_word["text"]),
                    x0=float(raw_word["x0"]),
                    x1=float(raw_word["x1"]),
                    top=float(raw_word["top"]),
                    bottom=float(raw_word["bottom"]),
                    page_number=page_number,
                )
                for raw_word in raw_words
                if normalize_spaces(raw_word["text"])
            ]

            rows = _build_rows(words, page_number=page_number, page_width=float(page.width))
            pages.append(
                PdfPage(
                    number=page_number,
                    width=float(page.width),
                    height=float(page.height),
                    words=words,
                    rows=rows,
                )
            )

    return PdfDocument(path=pdf_path, pages=pages)
