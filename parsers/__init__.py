from __future__ import annotations

from .cgd_movimentos import CgdMovimentosParser
from .generic_statement import GenericStatementParser
from .layout import PdfDocument, load_pdf_document
from .millennium_bcp import MillenniumBcpParser
from .novo_banco import NovoBancoParser
from .qonto import QontoParser
from .revolut import RevolutParser


PARSERS = [
    RevolutParser(),
    QontoParser(),
    NovoBancoParser(),
    CgdMovimentosParser(),
    MillenniumBcpParser(),
    GenericStatementParser(),
]


def select_parser(document: PdfDocument):
    for parser in PARSERS:
        if parser.can_parse(document):
            return parser
    raise ValueError(f"No parser matched document: {document.filename}")
