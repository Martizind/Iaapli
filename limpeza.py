import re
import pandas as pd

def limpar_descricao(texto):
    # 1. Converter para maiúsculas
    texto = texto.upper()
    
    # 2. Remover números longos (IDs/referências)
    texto = re.sub(r'\d{5,}', '', texto)
    
    # 3. Remover caracteres especiais
    texto = re.sub(r'[^A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ\s]', '', texto)
    
    # 4. Remover espaços extra
    texto = " ".join(texto.split())
    
    return texto

# Teste 
exemplo = "COMPRA POS 4829102934 CONTINENTE MATOSINHOS - 2024-05-10"
print(f"Antes: {exemplo}")
print(f"Depois: {limpar_descricao(exemplo)}")