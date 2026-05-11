from __future__ import annotations

import re


def limpar_descricao(texto: str) -> str:
    texto = texto.upper()
    texto = re.sub(r"\d{5,}", "", texto)
    texto = re.sub(r"[^A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ\s]", "", texto)
    texto = " ".join(texto.split())
    return texto


if __name__ == "__main__":
    exemplo = "COMPRA POS 4829102934 CONTINENTE MATOSINHOS - 2024-05-10"
    print(f"Antes: {exemplo}")
    print(f"Depois: {limpar_descricao(exemplo)}")
