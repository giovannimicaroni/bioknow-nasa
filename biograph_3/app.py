from flask import Flask, render_template, request
from pyvis.network import Network
import networkx as nx
import pickle

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Função para criar um grafo de exemplo baseado em uma busca
# Dentro do app.py

# Dentro do app.py

def create_graph(query=""):
    # Carrega o grafo a partir do arquivo .gpickle
    with open("grafo_keywords.gpickle", "rb") as f:
        g = pickle.load(f)

    # Cria a visualização com Pyvis
    net = Network(height="100vh", width="100%", bgcolor="#000000", font_color="white")
    
    # --- LÓGICA DE BUSCA E DESTAQUE ---
    
    # 1. Converte a query para minúsculas para uma busca não sensível a maiúsculas/minúsculas
    search_term = query.lower() if query else ""
    
    # 2. Encontra todos os nós que contêm o termo de busca
    matching_nodes = []
    if search_term:
        for node in g.nodes():
            if search_term in node.lower():
                matching_nodes.append(node)

    # 3. Adiciona os nós ao grafo do Pyvis com cores e tamanhos diferentes
    for node in g.nodes():
        title = node.replace(".pdf", "") # Remove o .pdf para um título mais limpo
        
        if node in matching_nodes:
            # Nó destacado (cor vermelha, tamanho maior)
            net.add_node(node, label=title, color='#ff6e54', size=30, title=f"<b>MATCH:</b><br>{title}")
        else:
            # Nó padrão (cor azul, tamanho normal)
            net.add_node(node, label=title, color='#87ceff', size=15, title=title)

    # 4. Adiciona todas as arestas (conexões)
    net.add_edges(g.edges())

    # Aplica as opções de física que já tínhamos
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -40000,
          "centralGravity": 0.05,
          "springLength": 150
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
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
    
    # Gera o HTML e adiciona JavaScript para eliminar scroll
    html = net.generate_html()
    
    # Adiciona JavaScript customizado para eliminar scroll
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Remove scroll do corpo da página
        document.body.style.overflow = 'hidden';
        document.documentElement.style.overflow = 'hidden';
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.height = '100vh';
        document.body.style.background = '#000000';
        
        // Remove qualquer scroll de containers
        var containers = document.querySelectorAll('div');
        containers.forEach(function(container) {
            container.style.overflow = 'hidden';
        });
    });
    </script>
    </body>
    """
    
    # Insere o JavaScript antes do fechamento do body
    html = html.replace('</body>', custom_js)
    
    return html

# Rota para a página inicial
@app.route('/')
def home():
    return render_template('index.html')


# Rota para a página do grafo
@app.route('/graph')
def graph_page():
    # Pega o termo de busca da URL (ex: /graph?query=mars)
    search_query = request.args.get('query', 'N/A')
    return render_template('graph.html', query=search_query)


# Rota de API que GERA e RETORNA o HTML do grafo
@app.route('/get_graph_data')
def get_graph_data():
    search_query = request.args.get('query', '')
    graph_html = create_graph(search_query)
    return graph_html


if __name__ == '__main__':
    app.run(debug=True)