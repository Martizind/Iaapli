from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pdfplumber
import ollama
import re
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Permite que o site HTML fale com este servidor

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ── Funções do teu projeto ──────────────────────────────────────

def limpar_descricao(texto):
    texto = texto.upper()
    texto = re.sub(r'\d{5,}', '', texto)
    texto = re.sub(r'[^A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ\s]', '', texto)
    texto = " ".join(texto.split())
    return texto

def classificar(descricao):
    try:
        response = ollama.chat(model='llama3', messages=[{
            'role': 'user',
            'content': f"Classifica em UMA única palavra (Alimentação, Transportes, Lazer, Saúde, Habitação, Subscrições, Receita, Outros) esta transação bancária: {descricao}. Responde APENAS com a palavra, sem mais nada."
        }])
        return response['message']['content'].strip()
    except:
        return 'Outros'

def extrair_transacoes(pdf_path):
    transacoes = []
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            tabelas = pagina.extract_tables()
            for tabela in tabelas:
                for linha in tabela:
                    if not linha:
                        continue
                    # Tenta detetar linhas com data e valor
                    linha_str = [str(c).strip() if c else '' for c in linha]
                    # Procura padrão de data (ex: 2026-03-01 ou 01/03/2026)
                    data = None
                    for cel in linha_str:
                        if re.search(r'\d{4}-\d{2}-\d{2}', cel):
                            data = re.search(r'\d{4}-\d{2}-\d{2}', cel).group()
                            break
                        if re.search(r'\d{2}/\d{2}/\d{4}', cel):
                            d = re.search(r'\d{2}/\d{2}/\d{4}', cel).group()
                            partes = d.split('/')
                            data = f"{partes[2]}-{partes[1]}-{partes[0]}"
                            break
                    if not data:
                        continue
                    # Procura valor numérico
                    valor = None
                    tipo = 'debit'
                    for cel in reversed(linha_str):
                        cel_clean = cel.replace('.', '').replace(',', '.').replace(' ', '').replace('€','')
                        try:
                            v = float(cel_clean)
                            if v != 0:
                                valor = v
                                tipo = 'credit' if v > 0 else 'debit'
                                break
                        except:
                            continue
                    if not valor:
                        continue
                    # Descrição é o campo mais longo
                    desc = max(linha_str, key=len)
                    desc = limpar_descricao(desc)
                    if len(desc) < 3:
                        continue
                    transacoes.append({
                        'date': data,
                        'description': desc,
                        'amount': valor,
                        'type': tipo,
                        'category': 'A processar'
                    })
    return transacoes

# ── Endpoints ───────────────────────────────────────────────────

@app.route('/status')
def status():
    """Verifica se o Ollama está a correr"""
    try:
        ollama.chat(model='llama3', messages=[{'role':'user','content':'ok'}])
        return jsonify({'status': 'ok', 'model': 'llama3'})
    except:
        return jsonify({'status': 'offline'}), 503

@app.route('/processar', methods=['POST'])
def processar():
    """Recebe o PDF, extrai e classifica as transações"""
    pdf_path = None

    # Caso 1: ficheiro enviado pelo browser
    if 'file' in request.files:
        file = request.files['file']
        pdf_path = 'temp_extrato.pdf'
        file.save(pdf_path)

    # Caso 2: caminho do ficheiro enviado como texto
    elif request.json and 'path' in request.json:
        pdf_path = request.json['path']

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF não encontrado'}), 400

    # Extrair transações
    transacoes = extrair_transacoes(pdf_path)

    if not transacoes:
        return jsonify({'error': 'Não foi possível extrair transações. Verifica o formato do PDF.'}), 400

    # Classificar com IA
    for t in transacoes:
        t['category'] = classificar(t['description'])

    # Resumo
    income  = sum(t['amount'] for t in transacoes if t['type'] == 'credit')
    expense = sum(abs(t['amount']) for t in transacoes if t['type'] == 'debit')

    # Limpar ficheiro temporário
    if os.path.exists('temp_extrato.pdf'):
        os.remove('temp_extrato.pdf')

    return jsonify({
        'transactions': transacoes,
        'summary': {'income': income, 'expense': expense, 'count': len(transacoes)}
    })

# ── Arrancar o servidor ──────────────────────────────────────────

if __name__ == '__main__':
    print("🚀 FinTrack backend a correr em http://127.0.0.1:5000")
    print("   Abre o index.html no browser e carrega um PDF!")
    app.run(debug=True, port=5000)
