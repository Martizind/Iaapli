from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pdfplumber
import ollama
import re
import os
import json
from datetime import date

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ═══════════════════════════════════════════════════════════════
#  UTILITÁRIOS & CONFIGURAÇÃO DA MEMÓRIA RAG
# ═══════════════════════════════════════════════════════════════

MESES_EN = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
            'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
MESES_PT = {'jan':'01','fev':'02','mar':'03','abr':'04','mai':'05','jun':'06',
            'jul':'07','ago':'08','set':'09','out':'10','nov':'11','dez':'12'}

def parse_data(texto):
    if not texto: return None
    t = str(texto).strip()
    m = re.search(r'(\d{1,2})[/\-\.](\d{2})[/\-\.](\d{4})', t)
    if m: return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', t)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r'^(\d{2})\.(\d{2})$', t)
    if m:
        ano = str(date.today().year)
        return f"{ano}-{m.group(1)}-{m.group(2)}"
    m = re.match(r'([A-Za-z]{3,})\s+(\d{1,2}),?\s+(\d{4})', t)
    if m:
        mes = MESES_EN.get(m.group(1).lower()[:3], '01')
        return f"{m.group(3)}-{mes}-{m.group(2).zfill(2)}"
    m = re.match(r'^(\d{2})/(\d{2})$', t)
    if m:
        ano = str(date.today().year)
        return f"{ano}-{m.group(2)}-{m.group(1)}"
    return None

def inferir_ano(texto):
    for y in ['2026','2025','2024','2023']:
        if y in texto:
            return y
    return str(date.today().year)

def parse_valor(texto):
    if not texto: return None
    t = str(texto).strip()
    t = re.sub(r'[€$£CHFfr\s]', '', t)
    if not t or t in ['-', '–', '']: return None
    negativo = t.startswith('-') or t.startswith('(')
    t = t.lstrip('-(').rstrip(')')
    if re.search(r'^\d{1,3}(\.\d{3})+,\d{1,2}$', t):
        t = t.replace('.', '').replace(',', '.')
    elif re.search(r'^\d{1,3}(,\d{3})+\.\d{1,2}$', t):
        t = t.replace(',', '')
    elif re.match(r'^\d+,\d{1,2}$', t):
        t = t.replace(',', '.')
    try:
        v = float(t)
        return -v if negativo else v
    except:
        return None

def limpar_desc(texto):
    t = str(texto).strip()
    t = re.sub(r'\d{7}/\d+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _limpar_para_rag(texto: str) -> str:
    """Normaliza texto para comparação RAG: maiúsculas, sem acentos, sem caracteres especiais."""
    import unicodedata
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _carregar_rag():
    """Carrega a memória RAG do CSV e devolve vectorizer + matriz TF-IDF."""
    import csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    caminho = os.path.join(os.path.dirname(__file__), "memoria_rag.csv")
    if not os.path.exists(caminho):
        return None, None, []

    entradas = []
    try:
        with open(caminho, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                desc = row.get("description", "")
                clean = row.get("clean_description", "")
                direction = row.get("direction", "")
                cat = row.get("categoria_correta", "Outros")
                texto = _limpar_para_rag(f"{desc} {clean} {direction}")
                entradas.append((texto, cat))
    except Exception as e:
        print(f"[RAG] Erro ao carregar memória: {e}")
        return None, None, []

    if not entradas:
        return None, None, []

    textos = [e[0] for e in entradas]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matriz = vec.fit_transform(textos)
    return vec, matriz, entradas

# Carregar RAG no arranque
_rag_vec, _rag_matriz, _rag_entradas = _carregar_rag()
print(f"[RAG] Memória carregada: {len(_rag_entradas)} entradas")

def _classificar_por_rag(descricao: str, direction: str, threshold: float = 0.30):
    """Procura a entrada mais parecida na memória RAG usando TF-IDF + cosine similarity."""
    if _rag_vec is None or _rag_matriz is None or not _rag_entradas:
        return None

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    query = _limpar_para_rag(f"{descricao} {direction}")
    query_vec = _rag_vec.transform([query])
    scores = cosine_similarity(query_vec, _rag_matriz).flatten()
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        cat = _rag_entradas[best_idx][1]
        print(f"[RAG] '{descricao[:40]}' → '{cat}' (score={best_score:.2f})")
        return cat
    return None

def classificar(descricao, cat_banco=None, direction="debit"):
    """Pipeline de classificação em 3 camadas: Regras -> RAG -> Llama 3"""
    desc_limpa = _limpar_para_rag(descricao)

    # ── Camada 1: Regras Estáticas Determinísticas ──────────────────
    mapa = {
        'carregar': 'Receita', 'câmbio': 'Câmbio', 'transferência': 'Transferência',
        'transfer': 'Transferência', 'pagserv': 'Outros', 'imposto': 'Impostos',
        'ordenado': 'Receita', 'salário': 'Receita', 'stripe': 'Receita',
    }
    if cat_banco:
        for k, v in mapa.items():
            if k in cat_banco.lower():
                print(f"[REGRAS] '{descricao[:40]}' → '{v}'")
                return v

    # ── Camada 2: Memória Local Dinâmica (RAG) ─────────────────────
    cat_rag = _classificar_por_rag(descricao, direction)
    if cat_rag:
        return cat_rag

    # ── Camada 3: Modelo Generativo Local (Llama 3) ────────────────
    prompt = (
        "Classifica esta transação bancária numa ÚNICA palavra das seguintes categorias: "
        "Alimentação, Transportes, Lazer, Saúde, Habitação, Subscrições, Receita, "
        "Educação, Viagens, Câmbio, Transferência, Impostos, Outros.\n"
        f"Transação: {descricao}\n"
        "Responde APENAS com a palavra, sem pontuação nem explicação."
    )
    try:
        r = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        cat = r["message"]["content"].strip().split("\n")[0].split()[0].rstrip(".")
        validas = ["Alimentação","Transportes","Lazer","Saúde","Habitação","Subscrições",
                   "Receita","Educação","Viagens","Câmbio","Transferência","Impostos","Outros"]
        result = cat if cat in validas else "Outros"
        print(f"[LLAMA3] '{descricao[:40]}' → '{result}'")
        return result
    except Exception as e:
        print(f"[LLAMA3] Erro: {e}")
        return "Outros"

# ═══════════════════════════════════════════════════════════════
#  DETECÇÃO DE BANCO (LÓGICA DOS EXTRATOS REAL)
# ═══════════════════════════════════════════════════════════════

def detectar_banco(texto):
    t = texto.lower()
    if 'revolut' in t:
        if 'dinheiro a entrar' in t or 'extrato personalizado' in t:
            return 'revolut_pt'
        return 'revolut_en'
    if 'millennium' in t or 'millenniumbcp' in t: return 'millennium'
    if 'caixa geral' in t or 'caixadirecta' in t: return 'cgd'
    if 'crédito agrícola' in t or 'credito agricola' in t: return 'ca'
    if 'qonto' in t: return 'qonto'
    if 'santander' in t: return 'santander'
    if 'bpi' in t: return 'bpi'
    if 'novobanco' in t: return 'novobanco'
    return 'generico'

# ═══════════════════════════════════════════════════════════════
#  PARSERS ESPECÍFICOS DO PDFPLUMBER
# ═══════════════════════════════════════════════════════════════

def parse_revolut_pt(paginas):
    txs = []
    ignorar = {'data','descrição','categoria','dinheiro','saldo','imposto','comissões',
                'total','extrato','informaç','revolut','página','contas','conta','investment',
                'cripto','gerado','outros','entrar/sair','resumos','operações','banco'}

    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                if not cells or not cells[0]: continue
                data = parse_data(cells[0])
                if not data: continue
                desc = cells[1].strip() if len(cells) > 1 else ''
                if not desc or any(ig in desc.lower() for ig in ignorar): continue
                if len(desc) < 3: continue
                desc = limpar_desc(desc)
                cat_banco = cells[2].strip() if len(cells) > 2 else ''
                valor_str = cells[3].strip() if len(cells) > 3 else ''
                valor_str = re.sub(r'[€\s]', '', valor_str)
                valor = parse_valor(valor_str)
                if valor is None or valor == 0: continue
                tipo = 'credit' if valor > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': valor,
                            'type': tipo, 'cat_banco': cat_banco})

    if not txs:
        for pagina in paginas:
            texto = pagina.extract_text()
            if not texto: continue
            linhas = texto.split('\n')
            for linha in linhas:
                linha = linha.strip()
                m = re.match(r'^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(Carregar|Outros|Câmbio|Comerciante|Transferência)\s+(-?[\d,]+)€\s+', linha)
                if not m: continue
                data = parse_data(m.group(1))
                if not data: continue
                desc = limpar_desc(m.group(2))
                cat_banco = m.group(3)
                valor = parse_valor(m.group(4))
                if not valor or valor == 0: continue
                tipo = 'credit' if valor > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': valor,
                            'type': tipo, 'cat_banco': cat_banco})
    return txs

def parse_revolut_en(paginas):
    txs = []
    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                if len(cells) < 3: continue
                data = parse_data(cells[0])
                if not data: continue
                desc = limpar_desc(cells[1]) if len(cells) > 1 else ''
                if not desc or len(desc) < 3: continue
                ignorar = {'date','description','money','balance','product'}
                if any(ig in desc.lower() for ig in ignorar): continue
                money_out = parse_valor(cells[2]) if len(cells) > 2 else None
                money_in  = parse_valor(cells[3]) if len(cells) > 3 else None
                if money_in and money_in > 0:
                    txs.append({'date': data, 'description': desc, 'amount': money_in,
                                'type': 'credit', 'cat_banco': ''})
                elif money_out and money_out > 0:
                    txs.append({'date': data, 'description': desc, 'amount': -money_out,
                                'type': 'debit', 'cat_banco': ''})

        if not txs:
            texto = pagina.extract_text()
            if not texto: continue
            linhas = texto.split('\n')
            for i, linha in enumerate(linhas):
                linha = linha.strip()
                m = re.match(r'^([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\s+(.+)', linha)
                if not m: continue
                data = parse_data(m.group(1))
                if not data: continue
                resto = m.group(2)
                vals_raw = re.findall(r'€[\d,\.]+', resto)
                vals = [parse_valor(v) for v in vals_raw if parse_valor(v) is not None]
                desc = re.sub(r'€[\d,\.]+', '', resto).strip()
                desc = limpar_desc(desc)
                if not desc or len(desc) < 2: continue
                prox = linhas[i+1].strip() if i+1 < len(linhas) else ''
                e_receita = prox.startswith('From:')
                if len(vals) >= 2:
                    v = vals[-2]
                elif len(vals) == 1:
                    v = vals[0]
                else:
                    continue
                tipo = 'credit' if (e_receita or v > 0) else 'debit'
                amount = abs(v) if tipo == 'credit' else -abs(v)
                txs.append({'date': data, 'description': desc, 'amount': amount,
                            'type': tipo, 'cat_banco': ''})
    return txs

def parse_millennium(paginas):
    txs = []
    for pagina in paginas:
        texto = pagina.extract_text()
        if not texto: continue
        ano = inferir_ano(texto)
        linhas = texto.split('\n')
        saldo_anterior = None

        for linha in linhas:
            linha = linha.strip()
            m_saldo = re.search(r'SALDO INICIAL\s+([\d\s]+\.\d{2})', linha)
            if m_saldo:
                try: saldo_anterior = float(m_saldo.group(1).replace(' ', ''))
                except: pass
                continue

            m = re.match(r'^(\d{2}\.\d{2})\s+(\d{2}\.\d{2})\s+(.+)', linha)
            if not m: continue

            data_str = m.group(1)
            partes = data_str.split('.')
            data = f"{ano}-{partes[1].zfill(2)}-{partes[0].zfill(2)}"
            resto = m.group(3)

            nums_raw = re.findall(r'\d{1,3}(?:\s\d{3})*\.\d{2}', resto)
            if not nums_raw: continue

            nums = []
            for n in nums_raw:
                try: nums.append(float(n.replace(' ', '')))
                except: pass
            if not nums: continue

            idx_num = resto.find(nums_raw[0])
            desc_raw = resto[:idx_num].strip()
            desc = limpar_desc(desc_raw)
            if not desc or len(desc) < 2: continue

            ignorar_desc = ['saldo','debito','credito','data','lanc','disponivel','inicial','final']
            if any(ig in desc.lower() for ig in ignorar_desc): continue

            saldo_atual = nums[-1]

            if saldo_anterior is not None:
                diff = round(saldo_atual - saldo_anterior, 2)
                tipo = 'credit' if diff > 0 else 'debit'
                amount = abs(diff)
            else:
                amount = nums[-2] if len(nums) >= 2 else nums[0]
                tipo = 'debit'

            saldo_anterior = saldo_atual
            if amount == 0: continue

            txs.append({'date': data, 'description': desc,
                        'amount': amount if tipo == 'credit' else -amount,
                        'type': tipo, 'cat_banco': ''})
    return txs

def parse_qonto(paginas):
    txs = []
    for pagina in paginas:
        texto = pagina.extract_text()
        if not texto: continue
        ano = inferir_ano(texto)
        linhas = [l.strip() for l in texto.split('\n')]

        i = 0
        while i < len(linhas):
            linha = linhas[i]

            m_data = re.match(r'^(\d{2}/\d{2})\s*(.*)', linha)
            if not m_data:
                i += 1
                continue

            data_raw = m_data.group(1)
            partes = data_raw.split('/')
            dia = partes[0].zfill(2)
            mes = partes[1].zfill(2)
            data = f"{ano}-{mes}-{dia}"
            desc_inicial = m_data.group(2).strip()

            desc_parts = [desc_inicial] if desc_inicial else []
            valor = None
            tipo = 'debit'
            j = i + 1

            while j < len(linhas) and j < i + 8:
                l = linhas[j]
                m_val = re.match(r'^([+\-])\s*([\d\s]+[\.,]\d{2})\s*(?:EUR|USD|ZAR|CHF)?', l)
                if m_val:
                    sinal = m_val.group(1)
                    v_str = m_val.group(2).replace(' ', '')
                    valor = parse_valor(v_str)
                    tipo = 'credit' if sinal == '+' else 'debit'
                    j += 1
                    break

                if re.match(r'^\d+\.\d+ \w+ = \d+\.\d+ \w+', l):
                    j += 1
                    continue

                if re.match(r'^Cartão \*\*', l) or re.match(r'^TECHNOLOGY EXPENSES', l):
                    j += 1
                    continue

                if re.match(r'^\d{2}/\d{2}\s', l) and j > i + 1:
                    break

                if l and len(l) > 1:
                    desc_parts.append(l)
                j += 1

            if valor is not None and valor != 0:
                desc = limpar_desc(' '.join(desc_parts))
                desc = re.sub(r'Cartão \*+\d+', '', desc).strip()
                if len(desc) > 1:
                    amount = abs(valor) if tipo == 'credit' else -abs(valor)
                    txs.append({'date': data, 'description': desc,
                                'amount': amount, 'type': tipo, 'cat_banco': ''})
            i = j
    return txs

def parse_cgd(paginas):
    txs = []
    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                if len(cells) < 4: continue
                data = parse_data(cells[0])
                if not data: continue
                desc = limpar_desc(cells[2]) if len(cells) > 2 else ''
                if not desc or len(desc) < 2: continue
                ignorar = {'data','descrição','montante','saldo','movimento','valor'}
                if any(ig in desc.lower() for ig in ignorar): continue
                valor = parse_valor(cells[3]) if len(cells) > 3 else None
                if valor is None or valor == 0: continue
                tipo = 'credit' if valor > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': valor,
                            'type': tipo, 'cat_banco': ''})
    return txs

def parse_ca(paginas):
    txs = []
    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                if len(cells) < 5: continue
                data = parse_data(cells[0])
                if not data: continue
                desc = limpar_desc(cells[2]) if len(cells) > 2 else ''
                if not desc or len(desc) < 2: continue
                tipo_str = cells[-1].lower() if cells[-1] else ''
                valor_str = cells[3] if len(cells) > 3 else ''
                valor = parse_valor(valor_str)
                if valor is None or valor == 0: continue
                valor = abs(valor)
                if 'créd' in tipo_str or 'cred' in tipo_str:
                    tipo = 'credit'; amount = valor
                else:
                    tipo = 'debit'; amount = -valor
                txs.append({'date': data, 'description': desc, 'amount': amount,
                            'type': tipo, 'cat_banco': ''})
    return txs

def parse_santander(paginas):
    txs = []
    ignorar = {'data','descrição','valor','saldo','movimento','extrato','santander',
               'resumo','total','banco','iban'}
    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                data = None
                for c in cells[:3]:
                    d = parse_data(c)
                    if d: data = d; break
                if not data: continue
                desc_cells = [c for c in cells if not parse_data(c) and parse_valor(c) is None and len(c) > 3]
                if not desc_cells: continue
                desc = limpar_desc(max(desc_cells, key=len))
                if not desc or any(ig in desc.lower() for ig in ignorar): continue
                vals = [(c, parse_valor(c)) for c in cells if parse_valor(c) is not None and parse_valor(c) != 0]
                if not vals: continue
                v = vals[-2][1] if len(vals) >= 2 else vals[-1][1]
                tipo = 'credit' if v > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': v,
                            'type': tipo, 'cat_banco': ''})
    return txs

def parse_generico(paginas):
    txs = []
    for pagina in paginas:
        tabelas = pagina.extract_tables()
        for tabela in tabelas:
            for linha in tabela:
                if not linha: continue
                cells = [str(c).strip() if c else '' for c in linha]
                data = None
                for c in cells:
                    data = parse_data(c)
                    if data: break
                if not data: continue
                vals = [(c, parse_valor(c)) for c in cells if parse_valor(c) is not None and parse_valor(c) != 0]
                desc_cells = [c for c in cells if not parse_data(c) and parse_valor(c) is None and len(c) > 3]
                if not desc_cells or not vals: continue
                desc = limpar_desc(max(desc_cells, key=len))
                if len(desc) < 3: continue
                v = vals[-2][1] if len(vals) >= 2 else vals[0][1]
                tipo = 'credit' if v > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': v,
                            'type': tipo, 'cat_banco': ''})

        if not txs:
            texto = pagina.extract_text()
            if not texto: continue
            for linha in texto.split('\n'):
                linha = linha.strip()
                if not linha: continue
                data = None
                for n in [10, 11, 12]:
                    data = parse_data(linha[:n])
                    if data: break
                if not data: continue
                nums = re.findall(r'-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}', linha)
                vals = [parse_valor(n) for n in nums if parse_valor(n) is not None and parse_valor(n) != 0]
                desc = re.sub(r'-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}', '', linha).strip()
                desc = re.sub(r'^\S{6,12}\s+', '', desc).strip()
                desc = limpar_desc(desc)
                if not desc or len(desc) < 3 or not vals: continue
                v = vals[0]
                tipo = 'credit' if v > 0 else 'debit'
                txs.append({'date': data, 'description': desc, 'amount': v,
                            'type': tipo, 'cat_banco': ''})
    return txs

# ═══════════════════════════════════════════════════════════════
#  ORQUESTRAÇÃO E PARSING PRINCIPAL
# ═══════════════════════════════════════════════════════════════

MAPA_PARSERS = {
    'revolut_pt': parse_revolut_pt,
    'revolut_en': parse_revolut_en,
    'millennium':  parse_millennium,
    'qonto':       parse_qonto,
    'cgd':         parse_cgd,
    'ca':          parse_ca,
    'santander':   parse_santander,
    'generico':    parse_generico,
}

def extrair_transacoes(pdf_path, banco_forcado=None):
    texto_total = ''
    paginas = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t: texto_total += t + '\n'
            paginas.append(p)

    banco = banco_forcado if banco_forcado else detectar_banco(texto_total)
    print(f"[FinTrack] Banco: {banco.upper()}")

    parser = MAPA_PARSERS.get(banco, parse_generico)
    txs = parser(paginas)

    if not txs and banco not in ['generico']:
        print(f"[FinTrack] Parser específico falhou, a tentar genérico...")
        txs = parse_generico(paginas)

    vistos = set()
    resultado = []
    for t in txs:
        chave = (t['date'], t['description'][:25], round(t['amount'], 2))
        if chave not in vistos:
            vistos.add(chave)
            t['category'] = 'A processar'
            resultado.append(t)

    resultado_valido = []
    import re as _re
    for t in resultado:
        d = str(t.get('date', '')).strip()
        if _re.match(r'^\d{4}-\d{2}-\d{2}$', d):
            partes = d.split('-')
            mes = int(partes[1])
            dia = int(partes[2])
            if 1 <= mes <= 12 and 1 <= dia <= 31:
                resultado_valido.append(t)
            else:
                print(f"[FinTrack] Data inválida ignorada: {d}")
        else:
            print(f"[FinTrack] Data com formato errado ignorada: {d!r}")

    resultado_valido.sort(key=lambda x: x['date'], reverse=True)
    print(f"[FinTrack] {len(resultado_valido)} transações extraídas ({len(resultado)-len(resultado_valido)} datas inválidas ignoradas)")
    return resultado_valido, banco

# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS DA API FLASK (INCLUI O CHAT IA INTEGRAÇÃO BRUTAL)
# ═══════════════════════════════════════════════════════════════

@app.route('/status')
def status():
    try:
        ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'ping'}])
        return jsonify({'status': 'ok', 'model': 'llama3', 'rag': len(_rag_entradas)})
    except Exception as e:
        return jsonify({'status': 'offline', 'error': str(e), 'rag': len(_rag_entradas)}), 503

@app.route('/bancos')
def bancos():
    return jsonify({'bancos': [
        {'id': 'auto',       'nome': '🔍 Detectar automaticamente'},
        {'id': 'revolut_pt', 'nome': 'Revolut (Português)'},
        {'id': 'revolut_en', 'nome': 'Revolut (Inglês)'},
        {'id': 'millennium', 'nome': 'Millennium BCP'},
        {'id': 'qonto',      'nome': 'Qonto'},
        {'id': 'cgd',        'nome': 'CGD Caixa Geral de Depósitos'},
        {'id': 'ca',         'nome': 'Crédito Agrícola'},
        {'id': 'santander',  'nome': 'Santander'},
        {'id': 'generico',   'nome': 'Outro banco (genérico)'},
    ]})

@app.route('/processar', methods=['POST'])
def processar():
    pdf_path = None
    banco_forcado = None

    if 'file' in request.files:
        file = request.files['file']
        pdf_path = 'temp_extrato.pdf'
        file.save(pdf_path)
        banco_forcado = request.form.get('banco') or None
    elif request.json and 'path' in request.json:
        pdf_path = request.json['path']
        banco_forcado = request.json.get('banco') or None

    if banco_forcado == 'auto':
        banco_forcado = None

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF não encontrado.'}), 400

    try:
        transacoes, banco = extrair_transacoes(pdf_path, banco_forcado)
    except Exception as e:
        return jsonify({'error': f'Erro ao ler PDF: {str(e)}'}), 500

    if not transacoes:
        return jsonify({
            'error': 'Não foi possível extrair transações.',
            'dica': 'Tenta selecionar o banco manualmente antes de processar.',
            'banco_detetado': banco
        }), 400

    print(f"[FinTrack] A classificar {len(transacoes)} transações (Regras → RAG → Llama3)...")
    for t in transacoes:
        t['category'] = classificar(
            t['description'],
            t.get('cat_banco', ''),
            t.get('type', 'debit')
        )
        t.pop('cat_banco', None)

    income  = sum(t['amount'] for t in transacoes if t['type'] == 'credit')
    expense = sum(abs(t['amount']) for t in transacoes if t['type'] == 'debit')

    if os.path.exists('temp_extrato.pdf'):
        os.remove('temp_extrato.pdf')

    return jsonify({
        'transactions': transacoes,
        'banco': banco,
        'summary': {
            'income':  round(income, 2),
            'expense': round(expense, 2),
            'count':   len(transacoes)
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'mensagem' not in data:
        return jsonify({'error': 'Sem mensagem'}), 400

    mensagem  = data.get('mensagem', '')
    historico = data.get('historico', [])
    contexto  = data.get('contexto', '')

    system = (
        "És um assistente financeiro pessoal inteligente e simpático chamado FinTrack AI. "
        "Respondes sempre em português de Portugal, de forma clara e direta. "
        "Tens acesso aos dados financeiros do utilizador e usas-os para dar respostas personalizadas. "
        "Quando relevante, menciona valores concretos dos dados. "
        "Nunca inventes dados que não foram fornecidos. "
        f"\n\nDados financeiros atuais do utilizador:\n{contexto}"
    )

    messages = [{'role': 'system', 'content': system}]
    for h in historico[:-1]:
        messages.append({'role': h['role'], 'content': h['content']})
    messages.append({'role': 'user', 'content': mensagem})

    try:
        r = ollama.chat(model='llama3', messages=messages)
        resposta = r['message']['content'].strip()
        return jsonify({'resposta': resposta})
    except Exception as e:
        return jsonify({'error': str(e), 'resposta': 'O modelo de IA não está disponível. Verifica se o Ollama está a correr.'}), 503

@app.route('/reclassificar', methods=['POST'])
def reclassificar():
    data = request.json
    if not data or 'category' not in data:
        return jsonify({'error': 'Dados inválidos'}), 400
    
    if 'description' in data and data['category']:
        desc = data['description']
        cat  = data['category']
        direction = data.get('direction', 'debit')
        novo_texto = _limpar_para_rag(f"{desc} {direction}")
        _rag_entradas.append((novo_texto, cat))
        print(f"[RAG] Nova entrada aprendida: '{desc[:40]}' → '{cat}'")
        
        global _rag_vec, _rag_matriz
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            textos = [e[0] for e in _rag_entradas]
            _rag_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
            _rag_matriz = _rag_vec.fit_transform(textos)
            print(f"[RAG] Vectorizer reconstruído com {len(_rag_entradas)} entradas")
        except Exception as e:
            print(f"[RAG] Erro ao reconstruir: {e}")

    return jsonify({'ok': True, 'category': data['category']})

@app.route('/sugestoes', methods=['POST'])
def sugestoes():
    data = request.json
    if not data or 'transactions' not in data:
        return jsonify({'error': 'Sem dados'}), 400

    txs = data['transactions']
    cats = {}
    income = 0
    for t in txs:
        if t['type'] == 'credit':
            income += t['amount']
        else:
            cats[t.get('category','Outros')] = cats.get(t.get('category','Outros'), 0) + abs(t['amount'])

    total = sum(cats.values())
    top = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:6]
    resumo = f"Receitas: €{income:.2f}\nDespesas: €{total:.2f}\nSaldo: €{income-total:.2f}\n\nPor categoria:\n"
    resumo += "\n".join(f"- {c}: €{v:.2f} ({v/total*100:.1f}%)" for c, v in top if total > 0)

    prompt = (
        "És consultor financeiro pessoal. Analisa este resumo e gera exatamente 4 sugestões "
        "práticas para poupar dinheiro no próximo mês, com valores estimados. "
        "Usa linguagem simples em português de Portugal. "
        "Formata cada sugestão como: '💡 Título: Explicação com valor estimado'\n\n"
        f"Dados financeiros:\n{resumo}"
    )
    try:
        r = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return jsonify({'sugestoes': r['message']['content'].strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║   FinTrack — Extrato Bancário Inteligente    ║")
    print("║   Regras → RAG → Llama 3                     ║")
    print(f"║   RAG: {len(_rag_entradas)} entradas carregadas{'':>17}║")
    print("║   http://127.0.0.1:5000                      ║")
    print("╚══════════════════════════════════════════════╝")
    app.run(debug=True, port=5000)