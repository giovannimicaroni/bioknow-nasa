from flask import Flask, render_template, request, jsonify
from pyvis.network import Network
import networkx as nx
import pickle
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)

# Cache global para o grafo HTML
graph_cache = {}

def create_graph(query="", filters=None):
    """
    Cria o grafo com busca e filtros
    
    Args:
        query: termo de busca para destacar nós
        filters: dict com filtros, ex: {'keywords': ['mars', 'biology']}
    """
    # Cria chave de cache considerando query e filtros
    filter_key = str(sorted(filters.items())) if filters else ""
    cache_key = f"{query.lower()}_{filter_key}"
    
    if cache_key in graph_cache:
        return graph_cache[cache_key]
    
    # Carrega o grafo completo
    with open("grafo_keywords.gpickle", "rb") as f:
        g = pickle.load(f)
    
    # Aplica filtros se existirem
    filtered_graph = g.copy()
    
    if filters and 'keywords' in filters and filters['keywords']:
        keywords = [k.lower() for k in filters['keywords']]
        
        # Filtra nós que contêm pelo menos uma das keywords no título OU nos atributos
        nodes_to_keep = set()
        for node in g.nodes():
            node_lower = node.lower()
            
            # Verifica no título do nó
            title_match = any(keyword in node_lower for keyword in keywords)
            
            # Verifica nos atributos do nó (keyword attribute)
            attribute_match = False
            node_data = g.nodes[node]
            
            # Verifica se existe atributo 'keywords' (que é um dicionário)
            if 'keywords' in node_data:
                keyword_attr = node_data['keywords']
                
                # Se for um dicionário (como no seu caso)
                if isinstance(keyword_attr, dict):
                    # Combina todas as chaves do dicionário em uma string
                    all_keyword_phrases = ' '.join(keyword_attr.keys()).lower()
                    attribute_match = any(keyword in all_keyword_phrases for keyword in keywords)
                
                # Se for uma string
                elif isinstance(keyword_attr, str):
                    keyword_attr_lower = keyword_attr.lower()
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
                
                # Se for uma lista
                elif isinstance(keyword_attr, list):
                    keyword_attr_lower = [k.lower() for k in keyword_attr]
                    attribute_match = any(keyword in ' '.join(keyword_attr_lower) for keyword in keywords)
            
            # Verifica também 'keyword' (singular) como fallback
            if 'keyword' in node_data and not attribute_match:
                keyword_attr = node_data['keyword']
                if isinstance(keyword_attr, str):
                    keyword_attr_lower = keyword_attr.lower()
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
                elif isinstance(keyword_attr, list):
                    keyword_attr_lower = [k.lower() for k in keyword_attr]
                    attribute_match = any(keyword in ' '.join(keyword_attr_lower) for keyword in keywords)
            
            # Verifica também no 'full_name' se existir
            if 'full_name' in node_data and not title_match and not attribute_match:
                full_name_lower = node_data['full_name'].lower()
                title_match = any(keyword in full_name_lower for keyword in keywords)
            
            # Adiciona o nó se houver match no título OU nos atributos
            if title_match or attribute_match:
                nodes_to_keep.add(node)
        
        # Remove nós que não passaram no filtro
        nodes_to_remove = set(g.nodes()) - nodes_to_keep
        filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # Se o grafo filtrado estiver vazio, retorna mensagem
    if len(filtered_graph.nodes()) == 0:
        return "<html><body style='background:#000;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui;'><h2>No results found for the selected filters</h2></body></html>"
    
    # Cria visualização
    net = Network(height="100vh", width="100%", bgcolor="#000000", font_color="white")
    
    # Busca e destaque
    search_term = query.lower() if query else ""
    matching_nodes = []
    
    if search_term:
        for node in filtered_graph.nodes():
            if search_term in node.lower():
                matching_nodes.append(node)
    
    # Adiciona nós com cores diferentes
    for node in filtered_graph.nodes():
        title = node.replace(".pdf", "")
        
        if node in matching_nodes:
            net.add_node(node, label=title, color='#ff6e54', size=30, title=f"<b>MATCH:</b><br>{title}")
        else:
            net.add_node(node, label=title, color='#87ceff', size=15, title=title)
    
    # Adiciona arestas
    net.add_edges(filtered_graph.edges())
    
    # Configurações de física
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.05,
          "springLength": 150
        },
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": false
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      },
      "configure": {
        "enabled": false
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)
    
    graph_html = net.generate_html()
    
    # CSS customizado
    custom_css = """
    <style>
        #loadingBar {
            display: none !important;
        }
        #mynetwork {
            background-color: #000000;
        }
    </style>
    """
    
    # JavaScript customizado
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        document.body.style.overflow = 'hidden';
        document.documentElement.style.overflow = 'hidden';
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.height = '100vh';
        document.body.style.background = '#000000';
        
        var containers = document.querySelectorAll('div');
        containers.forEach(function(container) {
            container.style.overflow = 'hidden';
        });
    });
    </script>
    </body>
    """
    
    graph_html = graph_html.replace('</head>', custom_css + '</head>')
    graph_html = graph_html.replace('</body>', custom_js)
    
    graph_cache[cache_key] = graph_html
    
    return graph_html


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/graph')
def graph_page():
    search_query = request.args.get('query', 'N/A')
    return render_template('graph.html', query=search_query)


@app.route('/get_graph_data')
def get_graph_data():
    search_query = request.args.get('query', '')
    
    # Pega filtros da query string
    filter_keywords = request.args.getlist('filter_keywords[]')
    
    filters = None
    if filter_keywords:
        filters = {'keywords': filter_keywords}
    
    graph_html = create_graph(search_query, filters)
    return graph_html


@app.route('/api/keywords')
def get_keywords():
    """API endpoint para obter lista de keywords disponíveis"""
    # Categorized keywords
    categorized_keywords = {
        "Biology": ['biology', 'cell', 'tissue', 'protein', 'gene', 'dna', 'organism'],
        "Space Environment": ['microgravity', 'radiation', 'space', 'iss', 'orbit'],
        "Life Sciences": ['plant', 'growth', 'metabolism', 'immune', 'cardiovascular'],
        "Human Health": ['astronaut', 'bone', 'muscle', 'sleep', 'exercise'],
        "Planetary": ['mars', 'moon', 'lunar', 'planetary']
    }
    
    # Return categorized structure
    return jsonify(categorized_keywords)

# No seu arquivo Python do Flask

def create_interactive_heatmap(top_n=30):
    """
    Cria um heatmap interativo com os 'top_n' nós mais conectados.
    """
    try:
        print("--- Iniciando create_interactive_heatmap ---")
        
        # 1. Carregar o grafo completo
        with open("grafo_keywords.gpickle", "rb") as f:
            g = pickle.load(f)
        print(f"Grafo carregado com {g.number_of_nodes()} nós.")
        
        if g.number_of_nodes() < top_n:
            top_n = g.number_of_nodes()

        # 2. Encontrar os 'top_n' nós
        top_nodes = sorted(g.degree, key=lambda item: item[1], reverse=True)[:top_n]
        top_node_names = [node for node, degree in top_nodes]
        print(f"Encontrados {len(top_node_names)} nós principais.")
        
        # 3. Criar subgrafo
        subgraph = g.subgraph(top_node_names)
        print("Subgrafo criado.")
        
        # 4. Calcular matriz de distância e similaridade
        dist_matrix = pd.DataFrame(nx.floyd_warshall_numpy(subgraph),
                                   index=subgraph.nodes(), columns=subgraph.nodes())
        similarity_matrix = 1 / (1 + dist_matrix)
        print("Matriz de similaridade calculada. Shape:", similarity_matrix.shape)
        
        # Encurtar os rótulos
        short_labels = [label.replace('.pdf', '')[:40] for label in similarity_matrix.index]
        similarity_matrix.index = short_labels
        similarity_matrix.columns = short_labels
        
        # 5. Criar a figura do Plotly
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix.values,
            x=similarity_matrix.columns,
            y=similarity_matrix.index,
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertemplate='Artigo 1: %{y}<br>Artigo 2: %{x}<br>Similaridade: %{z:.2f}<extra></extra>'
        ))
        print("Figura do Plotly criada.")
        
        # 6. Customizar o layout
        fig.update_layout(
            title=f'Heatmap de Similaridade entre os {top_n} Artigos Principais',
            template='plotly_dark',
            width=800,
            height=800,
            yaxis_scaleanchor='x',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )
        print("Layout da figura atualizado.")
        
        # 7. Converter para HTML
        heatmap_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        print("--- Finalizado: Convertido para HTML div com sucesso. ---")
        return heatmap_div

    except Exception as e:
        print(f"!!!!!! OCORREU UM ERRO DENTRO DE create_interactive_heatmap !!!!!!")
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        return "<h1>Ocorreu um erro ao gerar o gráfico. Verifique o terminal do servidor.</h1>"

# ... (suas rotas Flask existentes / e /graph etc.) ...

# ADICIONE ESTA NOVA ROTA NO FINAL DO SEU ARQUIVO
@app.route('/heatmap')
def heatmap_page():
    """
    Renderiza a página que exibe o heatmap interativo.
    """
    heatmap_div = create_interactive_heatmap()
    return render_template('heatmap.html', heatmap_div=heatmap_div)

if __name__ == '__main__':
    import os
    create_graph("")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
