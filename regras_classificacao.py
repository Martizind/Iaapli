from __future__ import annotations

import unicodedata


def remover_acentos(texto: str) -> str:
    normalized = unicodedata.normalize("NFKD", texto)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def classificar_por_regras(
    description: str,
    cleaned_description: str,
    details: str | None,
    direction: str,
) -> tuple[str, str] | None:
    description_ascii = remover_acentos(description).upper()
    cleaned_ascii = remover_acentos(cleaned_description).upper()
    details_ascii = remover_acentos(details or "").upper()
    combined_ascii = f"{description_ascii} {cleaned_ascii} {details_ascii}"

    if (
        description_ascii.startswith("TRF")
        or cleaned_ascii.startswith("TRF")
        or " MB WAY " in f" {description_ascii} "
        or " MBWAY " in f" {description_ascii} "
        or " MB WAY " in f" {cleaned_ascii} "
        or " MBWAY " in f" {cleaned_ascii} "
        or "TRANSFER TO " in description_ascii
        or "APPLE PAY DEPOSIT" in description_ascii
        or "REVOLUT BANK UAB SUCURSAL EM PORTUGAL" in description_ascii
        or "MEMORIA LIQUIDA LDA" in description_ascii
    ):
        return "Transferencias", "regra_trf_mbway"

    if "IMPOSTO" in description_ascii or "IMPOSTO" in cleaned_ascii:
        return "Impostos", "regra_imposto"

    if any(
        token in description_ascii
        for token in (
            "PAGSERV INSTITUTO GESTAO FINAN",
            "PAGSERV INSTITUTO REGISTOS",
            "PAG.IGCP",
        )
    ):
        return "Impostos", "regra_pagamento_estado"

    if (
        "CUSTO DE SERVICO" in description_ascii
        or "COM.MAN.CONTA" in description_ascii
        or "PACOTE M EMPRESA" in description_ascii
        or "SUBSCRICAO / COMISSOES" in details_ascii
        or cleaned_ascii.startswith("QONTO")
        or "IFTHENPAY" in description_ascii
    ):
        return "Taxas", "regra_taxa"

    if any(
        token in combined_ascii
        for token in (
            "CLAUDE.AI",
            "CHATGPT SUBSCR",
            "APPLE.COM/BILL",
            "WARP PRO SUB",
            "WINDSURF",
            "SUBSCRIPTION",
            "SUBSCR",
        )
    ):
        return "Subscricoes", "regra_subscricao"

    if any(
        token in description_ascii
        for token in ("MERCADONA", "ALDI", "LIDL", "CONTINENTE", "PINGO DOCE")
    ):
        return "Supermercado", "regra_supermercado"

    if any(
        token in description_ascii
        for token in (
            "MCDONALDS",
            "RESTAURANTE",
            "UBER * EATS",
            "HAMBURGUERIA",
            "BODEGAO",
            "CERV DIO",
            "EUREST",
        )
    ):
        return "Restauracao", "regra_restauracao"

    if any(token in description_ascii for token in ("GALP", "BP", "REPSOL", "E.S. ", "UBER")):
        return "Transportes", "regra_transportes"

    if any(
        token in description_ascii
        for token in ("AMZN ", "AMAZON", "TICKETNOTIFY")
    ):
        return "Compras", "regra_compras"

    if any(
        token in description_ascii
        for token in (
            "OPENAI",
            "CLAUDE",
            "NAME-CHEAP",
            "ADSELLR",
            "YADAPHONE",
            "PC DIGA",
            "WORTEN",
        )
    ):
        return "Tecnologia", "regra_tecnologia"

    if direction == "credit" and any(
        token in combined_ascii
        for token in ("STRIPE", "P/O MEMORIA", "ORD.PGT.DO ESTRG", "TECHNOLOGY EXPENSES", "TICKETORO")
    ):
        return "Rendimentos", "regra_credito"

    return None
