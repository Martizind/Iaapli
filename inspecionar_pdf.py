import pdfplumber

# Coloca aqui o nome do ficheiro do teu amigo
pdf_path = "C:\\Users\\Pfelix\\Desktop\\Proj_AI_Banco\\account-statement_2026-03-01_2026-03-31_en-us_f7075e.pdf" 

print(f"--- A iniciar extração de: {pdf_path} ---\n")

with pdfplumber.open(pdf_path) as pdf:
    # Vamos analisar apenas as primeiras 2 páginas para não inundar o terminal
    for i, pagina in enumerate(pdf.pages[:2]):
        print(f"=== PÁGINA {i+1} ===")
        
        # Opção A: Texto Bruto (bom para ver cabeçalhos e rodapés)
        texto = pagina.extract_text()
        print("\n[TEXTO BRUTO]:")
        print(texto[:500] + "...") # Primeiras 500 letras
        
        
        
        print("\n" + "="*30 + "\n")