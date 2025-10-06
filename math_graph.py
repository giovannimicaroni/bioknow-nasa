import os
import json
import pickle
import itertools
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tika import parser
import tika

tika.initVM()

from keybert import KeyBERT
from transformers import AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ================================================================
# 1. ETAPA DE EXTRAÇÃO DE TEXTO DOS PDFs
# ================================================================
def extract_pdfs_to_jsonl(pdf_folder: str, output_file: str):
    """Extrai texto de todos os PDFs de uma pasta e salva em JSONL"""
    
    texts = []

    print("📄 Extraindo textos dos PDFs...")
    for pdf in tqdm(os.listdir(pdf_folder)):
        if not pdf.lower().endswith(".pdf"):
            continue
        path_pdf = os.path.join(pdf_folder, pdf)
        parsed_pdf = parser.from_file(path_pdf)
        data = parsed_pdf.get("content", "")
        texts.append({"article": pdf, "text": data})

    with open(output_file, "w", encoding="utf-8") as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Textos extraídos e salvos em: {output_file}")
    return texts


# ================================================================
# 2. ETAPA DE EXTRAÇÃO DE KEYWORDS
# ================================================================
def extract_keywords(input_jsonl: str, output_jsonl: str):
    """Extrai keywords de cada texto e salva incrementalmente em JSONL"""
    embed_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", dtype="auto")
    kw_model = KeyBERT(model=embed_model)

    print("🧩 Extraindo keywords...")
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            data = json.loads(line)
            keyws = kw_model.extract_keywords(
                data["text"],
                keyphrase_ngram_range=(0, 2),
                stop_words="english",
                top_n=10,
                use_mmr=True,
                diversity=0.3
            )
            data["keywords"] = keyws
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Keywords extraídas e salvas em: {output_jsonl}")


# ================================================================
# 3. CLASSE DE ANÁLISE
# ================================================================
class KeywordAnalyzer:
    def _init_(self, jsonl_path, embed_model_name='all-MiniLM-L6-v2'):
        self.jsonl_path = jsonl_path
        self.data = []
        self.embed_model = SentenceTransformer(embed_model_name)
        self.load_data()

    def load_data(self):
        print("📚 Carregando dados...")
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"✓ {len(self.data)} artigos carregados")

    # ------------------------------------------------------------
    # ESTATÍSTICAS BÁSICAS
    # ------------------------------------------------------------
    def keyword_frequency_analysis(self, top_n=20):
        keyword_counts = Counter()
        keyword_in_texts = defaultdict(int)

        for item in self.data:
            text = item.get('text', '').lower()
            kws = [kw[0] for kw in item.get('keywords', [])]
            for kw in kws:
                keyword_counts[kw] += 1
                keyword_in_texts[kw] += text.count(kw.lower())

        print(f"\nTop {top_n} keywords mais comuns:")
        for kw, count in keyword_counts.most_common(top_n):
            pct = (count / len(self.data)) * 100
            avg_freq = keyword_in_texts[kw] / count
            print(f"  {kw:30s} | {count:3d} artigos ({pct:5.1f}%) | ~{avg_freq:.1f} vezes/artigo")

        return keyword_counts, keyword_in_texts

    # ------------------------------------------------------------
    # ANÁLISE SEMÂNTICA
    # ------------------------------------------------------------
    def calculate_article_embeddings(self):
        texts = []
        articles = []

        for item in tqdm(self.data, desc="Gerando embeddings"):
            text = item.get('text', '')
            words = text.split()[:1000]  # limitar tamanho
            texts.append(' '.join(words))
            articles.append(item['article'])

        embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        return embeddings, articles

    def similarity_analysis(self, embeddings, articles, top_n=10):
        sim_matrix = cosine_similarity(embeddings)
        similarities = []
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                similarities.append({
                    'article1': articles[i],
                    'article2': articles[j],
                    'similarity': sim_matrix[i, j]
                })
        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        print(f"\nTop {top_n} pares mais similares:")
        for sim in similarities[:top_n]:
            print(f"  {sim['similarity']:.3f} | {sim['article1']} <-> {sim['article2']}")
        return sim_matrix, similarities

    # ------------------------------------------------------------
    # VISUALIZAÇÕES
    # ------------------------------------------------------------
    def plot_top_keywords(self, keyword_counts, top_n=20):
        top_kws = dict(keyword_counts.most_common(top_n))
        plt.figure(figsize=(12, 8))
        plt.barh(list(top_kws.keys()), list(top_kws.values()))
        plt.gca().invert_yaxis()
        plt.xlabel("Número de Artigos")
        plt.title(f"Top {top_n} Keywords")
        plt.tight_layout()
        plt.savefig("outputs/top_keywords.png", dpi=300)
        print("✓ Gráfico salvo: outputs/top_keywords.png")

    def plot_similarity_heatmap(self, sim_matrix, articles):
        n_show = min(30, len(articles))
        plt.figure(figsize=(14, 12))
        sns.heatmap(sim_matrix[:n_show, :n_show],
                    xticklabels=[a[:30] for a in articles[:n_show]],
                    yticklabels=[a[:30] for a in articles[:n_show]],
                    cmap="YlOrRd", vmin=0, vmax=1)
        plt.title("Similaridade entre Artigos")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig("outputs/similarity_heatmap.png", dpi=300)
        print("✓ Gráfico salvo: outputs/similarity_heatmap.png")

    # ------------------------------------------------------------
    # PIPELINE COMPLETA
    # ------------------------------------------------------------
    def run_full_analysis(self, n_clusters=5):
        keyword_counts, _ = self.keyword_frequency_analysis()
        embeddings, articles = self.calculate_article_embeddings()
        sim_matrix, _ = self.similarity_analysis(embeddings, articles)
        self.plot_top_keywords(keyword_counts)
        self.plot_similarity_heatmap(sim_matrix, articles)

        results = {
            "top_keywords": dict(keyword_counts.most_common(50)),
        }

        # Converter também quaisquer np.int32 dentro das listas
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        results = convert_numpy(results)

        with open("outputs/analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("✓ Resultados salvos em outputs/analysis_results.json")
        return results


# ================================================================
# 4. CRIAÇÃO DO GRAFO DE CONEXÕES ENTRE ARTIGOS
# ================================================================
def create_keyword_graph(jsonl_path, output_file="outputs/grafo_keywords.gpickle"):
    artigos = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            keywords = [kw for kw, _ in data["keywords"]]
            artigos.append({"article": data["article"], "keywords": set(keywords)})

    G = nx.Graph()
    for art in artigos:
        short_name = art["article"][:25]
        G.add_node(short_name, full_name=art["article"], keywords=art["keywords"])

    for a, b in itertools.combinations(artigos, 2):
        intersec = a["keywords"].intersection(b["keywords"])
        if intersec:
            G.add_edge(a["article"][:25], b["article"][:25], weight=len(intersec), interseccao=intersec)

    pos = nx.spring_layout(G, seed=42, k=2)
    plt.figure(figsize=(18, 12))
    nx.draw_networkx(G, pos, with_labels=False, node_size=500, node_color="skyblue", alpha=0.8)
    plt.title("Rede de Artigos por Keywords Compartilhadas")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/graph_keywords.png", dpi=300)
    print("✓ Grafo salvo: outputs/graph_keywords.png")

    with open(output_file, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"✅ Grafo salvo como {output_file}")
    return G


if _name_ == "_main_":
    os.makedirs("outputs", exist_ok=True)
    pdf_folder = "Nasa Space Apps Livros 2"
    raw_texts_jsonl = "outputs/textos.jsonl"
    keywords_jsonl = "outputs/keywords_resultados.jsonl"

    # 1. Extrair textos
    if not os.path.exists(raw_texts_jsonl):
        extract_pdfs_to_jsonl(pdf_folder, raw_texts_jsonl)

    # 2. Extrair keywords
    if not os.path.exists(keywords_jsonl):
        extract_keywords(raw_texts_jsonl, keywords_jsonl)

    # 3. Analisar resultados
    analyzer = KeywordAnalyzer(keywords_jsonl)
    results = analyzer.run_full_analysis(n_clusters=5)

    # 4. Criar grafo de conexões
    G = create_keyword_graph(keywords_jsonl)