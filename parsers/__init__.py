from __future__ import annotations

from .layout import PdfDocument, load_pdf_document
from .millennium_bcp import MillenniumBcpParser
from .qonto import QontoParser
from .revolut import RevolutParser


PARSERS = [
    RevolutParser(),
    QontoParser(),
    MillenniumBcpParser(),
]


def select_parser(document: PdfDocument):
    for parser in PARSERS:
        if parser.can_parse(document):
            return parser
    raise ValueError(f"No parser matched document: {document.filename}")
