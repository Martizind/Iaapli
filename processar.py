import ollama

# Uma lista de transações que poderiam vir do teu CSV
transacoes = [
    "POSTO GALP PORTO",
    "RESTAURANTE O MANEL",
    "MEO TELECOMUNICACOES",
    "LIDL GAIA"
]

print("--- A iniciar classificação com Llama 3 ---")

for t in transacoes:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f"Classifica em uma única palavra (Ex: Transportes, Alimentação, Lazer, Saúde) esta transação: {t}",
        },
    ])
    categoria = response['message']['content'].strip()
    print(f"Transação: {t} | Categoria: {categoria}")