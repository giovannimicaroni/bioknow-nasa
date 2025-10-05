from flask import Flask, render_template, request, jsonify, session, send_file, abort
from flask_cors import CORS
from pyvis.network import Network
import networkx as nx
import pickle
import plotly.graph_objects as go
import pandas as pd
import requests
import os
import uuid
import json
import re
import csv
import subprocess
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from sklearn.metrics.pairwise import cosine_similarity
import random 


# AI Provider imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Enable CORS for external integrations
CORS(app, resources={
    r"/ask-lumi/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure articles data
ARTICLES_TXT_FOLDER = 'data/processed'
ARTICLES_PDF_FOLDER = 'data/raw'
ARTICLES_MAPPING_FILE = 'data/articles_mapping.tsv'

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('sessions', exist_ok=True)

# LM Studio configuration for WSL -> Windows
def get_windows_host_ip():
    """Descobrir IP do Windows host a partir do WSL"""
    try:
        result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
        if result.stdout:
            gateway = result.stdout.split('via')[1].split()[0] if 'via' in result.stdout else None
            if gateway:
                return gateway
    except:
        pass
    return '172.28.160.1'

windows_ip = get_windows_host_ip()
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', f'http://{windows_ip}:3000/v1/chat/completions')
LM_STUDIO_MODEL = os.getenv('LM_STUDIO_MODEL', 'deepseek/deepseek-r1-0528-qwen3-8b')

print(f"🔗 LM Studio URL: {LM_STUDIO_URL}")
print(f"🤖 Model: {LM_STUDIO_MODEL}")

# Cache global para o grafo HTML
graph_cache = {}

# ============================================================================
# AWS S3 FUNCTIONS
# ============================================================================

def get_s3_pdf_url(pdf_filename):
    """
    Gera URL do S3 para um PDF
    """
    # Remove espaços e caracteres especiais, substitui por %20
    encoded_filename = pdf_filename.replace(' ', '%20')
    s3_url = f"https://nasa-spaceapps-25.s3.us-east-1.amazonaws.com/{encoded_filename}"
    return s3_url

# ============================================================================
# HELPER FUNCTIONS - BioKnowdes
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, filename):
    """Extract text from uploaded files"""
    text = ""
    
    if filename.endswith('.txt') or filename.endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif filename.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
    
    elif filename.endswith('.docx'):
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    
    return text

def truncate_text_for_context(text, max_chars=8000):
    """Truncate text to fit within token limits"""
    if len(text) <= max_chars:
        return text
    
    first_part = text[:max_chars//2]
    last_part = text[-max_chars//2:]
    
    return f"{first_part}\n\n[... conteúdo truncado ...]\n\n{last_part}"

def load_articles_mapping():
    """Load the articles mapping from TSV file"""
    mapping = {}
    
    if not os.path.exists(ARTICLES_MAPPING_FILE):
        return mapping
    
    with open(ARTICLES_MAPPING_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            txt_filename = row.get('txt_filename', '')
            if txt_filename:
                mapping[txt_filename] = {
                    'title': row.get('title', ''),
                    'pdf_filename': row.get('pdf_filename', ''),
                    'download_url': row.get('download_url', ''),
                    'pmc_link': row.get('pmc_link', '')
                }
    
    return mapping

def load_article_content(filename):
    """Load article content from processed folder"""
    file_path = os.path.join(ARTICLES_TXT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

def search_articles_by_keywords(keywords):
    """Search articles by keywords in filenames"""
    if not os.path.exists(ARTICLES_TXT_FOLDER):
        return []
    
    results = []
    
    if not keywords:
        return results
    
    for filename in os.listdir(ARTICLES_TXT_FOLDER):
        if filename.endswith('_text.txt'):
            filename_lower = filename.lower()
            if any(keyword.lower() in filename_lower for keyword in keywords):
                results.append(filename)
    
    return results

# AI Provider Functions
def process_ai_response(ai_response):
    """Process AI response to extract thinking and final content"""
    thinking_content = ""
    final_response = ai_response.strip()
    
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, ai_response, re.DOTALL)
    
    if think_matches:
        thinking_content = think_matches[0].strip()
        final_response = re.sub(think_pattern, '', ai_response, flags=re.DOTALL).strip()
    
    return jsonify({
        'response': final_response,
        'thinking': thinking_content,
        'has_thinking': bool(thinking_content)
    })

def call_openai_api(messages, settings):
    """Call OpenAI API"""
    if not openai:
        raise Exception("OpenAI library not installed")
    
    openai_settings = settings.get('openai', {})
    api_key = openai_settings.get('api_key', os.getenv('OPENAI_API_KEY'))
    model = openai_settings.get('model', 'gpt-4')
    
    if not api_key:
        raise Exception("OpenAI API key not configured")
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return response.choices[0].message.content

def call_anthropic_api(messages, settings):
    """Call Anthropic API"""
    if not anthropic:
        raise Exception("Anthropic library not installed")
    
    anthropic_settings = settings.get('anthropic', {})
    api_key = anthropic_settings.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
    model = anthropic_settings.get('model', 'claude-3-5-sonnet-20241022')
    
    if not api_key:
        raise Exception("Anthropic API key not configured")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    anthropic_messages = []
    system_message = ""
    
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        else:
            anthropic_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_message if system_message else anthropic.NOT_GIVEN,
        messages=anthropic_messages
    )
    
    return response.content[0].text

def call_gemini_api(messages, settings):
    """Call Google Gemini API"""
    if not genai:
        raise Exception("Google Generative AI library not installed")
    
    gemini_settings = settings.get('gemini', {})
    api_key = gemini_settings.get('api_key', os.getenv('GEMINI_API_KEY'))
    model_name = gemini_settings.get('model', 'gemini-1.5-pro')
    
    if not api_key:
        raise Exception("Gemini API key not configured")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    gemini_messages = []
    for msg in messages:
        role = "user" if msg['role'] in ['user', 'system'] else "model"
        gemini_messages.append({
            "role": role,
            "parts": [msg['content']]
        })
    
    chat = model.start_chat(history=gemini_messages[:-1])
    response = chat.send_message(gemini_messages[-1]['parts'][0])
    
    return response.text

def call_lm_studio_api(messages):
    """Call LM Studio local API"""
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            raise Exception(f"LM Studio API error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error connecting to LM Studio: {str(e)}")

# ============================================================================
# GRAPH FUNCTIONS - Original Project
# ============================================================================

def create_graph(query="", filters=None):
    """
    Creates graph with search and filters
    
    Args:
        query: search term to highlight nodes
        filters: dict with filters, e.g., {'keywords': ['mars', 'biology']}
    """
    filter_key = str(sorted(filters.items())) if filters else ""
    cache_key = f"{query.lower()}_{filter_key}"
    
    if cache_key in graph_cache:
        return graph_cache[cache_key]
    
    with open("grafo_keywords.gpickle", "rb") as f:
        g = pickle.load(f)
    
    filtered_graph = g.copy()
    
    # Apply keyword filters if present
    if filters and 'keywords' in filters and filters['keywords']:
        keywords = [k.lower().strip() for k in filters['keywords']]
        
        nodes_to_keep = set()
        for node in g.nodes():
            node_lower = node.lower()
            
            # Check if keyword matches node title/name
            title_match = any(keyword in node_lower for keyword in keywords)
            
            # Check node attributes for keyword matches
            attribute_match = False
            node_data = g.nodes[node]
            
            # Check 'keywords' attribute
            if 'keywords' in node_data:
                keyword_attr = node_data['keywords']
                
                if isinstance(keyword_attr, dict):
                    # Join all keys and check
                    all_keyword_phrases = ' '.join(str(k).lower() for k in keyword_attr.keys())
                    attribute_match = any(keyword in all_keyword_phrases for keyword in keywords)
                
                elif isinstance(keyword_attr, str):
                    keyword_attr_lower = keyword_attr.lower()
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
                
                elif isinstance(keyword_attr, list):
                    # Join all list items and check
                    keyword_attr_lower = ' '.join(str(k).lower() for k in keyword_attr)
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
            
            # Check 'keyword' attribute (singular)
            if 'keyword' in node_data and not attribute_match:
                keyword_attr = node_data['keyword']
                if isinstance(keyword_attr, str):
                    keyword_attr_lower = keyword_attr.lower()
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
                elif isinstance(keyword_attr, list):
                    keyword_attr_lower = ' '.join(str(k).lower() for k in keyword_attr)
                    attribute_match = any(keyword in keyword_attr_lower for keyword in keywords)
            
            # Check 'full_name' attribute
            if 'full_name' in node_data and not title_match and not attribute_match:
                full_name_lower = str(node_data['full_name']).lower()
                title_match = any(keyword in full_name_lower for keyword in keywords)
            
            # Keep node if any match found
            if title_match or attribute_match:
                nodes_to_keep.add(node)
        
        # Remove nodes that don't match
        nodes_to_remove = set(g.nodes()) - nodes_to_keep
        filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # Handle empty results
    if len(filtered_graph.nodes()) == 0:
        return """
        <html>
        <body style='background:#000;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui;flex-direction:column;'>
            <h2>No results found for the selected filters</h2>
            <p style='color:#888;margin-top:1rem;'>Try different keywords or clear filters</p>
        </body>
        </html>
        """
    
    # Create visualization network
    net = Network(height="100vh", width="100%", bgcolor="#000000", font_color="white")
    
    search_term = query.lower() if query else ""
    matching_nodes = []
    
    if search_term:
        for node in filtered_graph.nodes():
            if search_term in node.lower():
                matching_nodes.append(node)
    
    # Add nodes to network
    for node in filtered_graph.nodes():
        title = node.replace(".pdf", "")
        node_data = filtered_graph.nodes[node]
        
        # Build tooltip with title and keywords
        tooltip_parts = [f"<div style='max-width:400px;'><b style='font-size:14px;color:#87ceff;'>{title}</b>"]
        
        # Extract keywords from node
        keywords_list = []
        if 'keywords' in node_data:
            keyword_attr = node_data['keywords']
            if isinstance(keyword_attr, dict):
                keywords_list = list(keyword_attr.keys())
            elif isinstance(keyword_attr, str):
                keywords_list = [keyword_attr]
            elif isinstance(keyword_attr, list):
                keywords_list = keyword_attr
        elif 'keyword' in node_data:
            keyword_attr = node_data['keyword']
            if isinstance(keyword_attr, str):
                keywords_list = [keyword_attr]
            elif isinstance(keyword_attr, list):
                keywords_list = keyword_attr
        
        # Add keywords to tooltip if they exist
        if keywords_list:
            tooltip_parts.append("<div style='margin-top:8px;padding-top:8px;border-top:1px solid #444;'>")
            tooltip_parts.append("<b style='color:#ff6e54;font-size:12px;'>Keywords:</b>")
            tooltip_parts.append("<ul style='margin:4px 0;padding-left:20px;font-size:11px;line-height:1.4;'>")
            for kw in keywords_list[:10]:  # Limit to 10 keywords
                tooltip_parts.append(f"<li>{kw}</li>")
            if len(keywords_list) > 10:
                tooltip_parts.append(f"<li><i>... and {len(keywords_list)-10} more keywords</i></li>")
            tooltip_parts.append("</ul></div>")
        
        tooltip_parts.append("</div>")
        tooltip_html = "".join(tooltip_parts)
        
        # Highlight matching nodes
        if node in matching_nodes:
            tooltip_html = f"<div style='max-width:400px;'><b style='color:#ff6e54;'>🔍 MATCH</b><br>{tooltip_html}</div>"
            net.add_node(node, label=title, color='#ff6e54', size=30, title=tooltip_html)
        else:
            net.add_node(node, label=title, color='#87ceff', size=15, title=tooltip_html)
    
    # Add edges
    net.add_edges(filtered_graph.edges())
    
    # Set physics options
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
    
    # Generate HTML
    graph_html = net.generate_html()
    
    # Add custom CSS
    custom_css = """
    <style>
        #loadingBar {
            display: none !important;
        }
        #mynetwork {
            background-color: #000000;
        }
        .vis-tooltip {
            background-color: #1a1a1a !important;
            border: 1px solid #444 !important;
            color: #fff !important;
            font-family: system-ui, -apple-system, sans-serif !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
            padding: 12px !important;
            border-radius: 6px !important;
        }
    </style>
    """
    
    # Add custom JS
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
    
    # Cache the result
    graph_cache[cache_key] = graph_html
    
    return graph_html

# Sua função, agora revisada
def create_interactive_heatmap_cosine(top_n=30, gpickle_path="grafo_keywords.gpickle"):
    """
    Cria um heatmap interativo de similaridade de cosseno entre 'top_n' nós 
    selecionados aleatoriamente de um pool dos mais conectados.
    """
    try:
        # Carrega o grafo
        with open(gpickle_path, "rb") as f:
            g = pickle.load(f)
        print(f"Grafo carregado com {g.number_of_nodes()} nós e {g.number_of_edges()} arestas.")

        # Ajusta top_n se houver menos nós
        if g.number_of_nodes() < top_n:
            top_n = g.number_of_nodes()

        # --- INÍCIO DA MODIFICAÇÃO ---
        # 1. Defina o tamanho do grupo de onde vamos sortear. 100 é um bom número.
        pool_size = 100
        if g.number_of_nodes() < pool_size:
            pool_size = g.number_of_nodes()

        # 2. Seleciona o grupo maior de nós com maior grau
        top_nodes_pool = sorted(g.degree, key=lambda item: item[1], reverse=True)[:pool_size]
        top_node_names_pool = [node for node, _ in top_nodes_pool]

        # 3. Sorteia 'top_n' nós DE DENTRO desse grupo
        top_node_names = random.sample(top_node_names_pool, top_n)
        
        print(f"{len(top_node_names)} nós principais selecionados aleatoriamente de um pool de {pool_size}.")
        # --- FIM DA MODIFICAÇÃO ---


        # O restante da função continua exatamente igual
        subgraph = g.subgraph(top_node_names)
        print("Subgrafo criado com sucesso.")

        adj_matrix = nx.to_numpy_array(subgraph, nodelist=top_node_names)
        print("Matriz de adjacência criada. Shape:", adj_matrix.shape)

        cosine_sim = cosine_similarity(adj_matrix)
        print("Matriz de similaridade de cosseno calculada.")
        
        similarity_matrix = pd.DataFrame(cosine_sim, index=top_node_names, columns=top_node_names)

        short_labels = [label.replace('.pdf', '')[:40] for label in similarity_matrix.index]
        similarity_matrix.index = short_labels
        similarity_matrix.columns = short_labels

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix.values,
            x=similarity_matrix.columns,
            y=similarity_matrix.index,
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertemplate='Artigo 1: %{y}<br>Artigo 2: %{x}<br>Similaridade: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Heatmap de Similaridade de Cosseno (Top {top_n} Artigos Amostrados)',
            template='plotly_dark',
            width=800,
            height=800,
            yaxis_scaleanchor='x',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        print("Heatmap de cosseno criado com sucesso.")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print("!!!!!! ERRO em create_interactive_heatmap_cosine !!!!!!")
        print("Erro:", e)
        import traceback
        traceback.print_exc()
        return "<h1>Erro ao gerar o heatmap de similaridade de cosseno. Verifique o log do servidor.</h1>"

# Sua rota Flask não precisa de nenhuma alteração
@app.route('/heatmap_cosine')
def heatmap_cosine_page():
    """
    Rota Flask para exibir o heatmap de similaridade de cosseno.
    """
    heatmap_div = create_interactive_heatmap_cosine()
    return render_template('heatmap.html', heatmap_div=heatmap_div)


# ============================================================================
# ROUTES - Original Project (Graph/Heatmap)
# ============================================================================

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
    
    filter_keywords = request.args.getlist('filter_keywords[]')
    
    filters = None
    if filter_keywords:
        filters = {'keywords': filter_keywords}
    
    graph_html = create_graph(search_query, filters)
    return graph_html

@app.route('/api/keywords')
def get_keywords():
    """API endpoint para obter lista de keywords disponíveis"""
    categorized_keywords = {
        "Biology": ['biology', 'cell', 'tissue', 'protein', 'gene', 'dna', 'organism'],
        "Space Environment": ['microgravity', 'radiation', 'space', 'iss', 'orbit'],
        "Life Sciences": ['plant', 'growth', 'metabolism', 'immune', 'cardiovascular'],
        "Human Health": ['astronaut', 'bone', 'muscle', 'sleep', 'exercise'],
        "Planetary": ['mars', 'moon', 'lunar', 'planetary']
    }
    
    return jsonify(categorized_keywords)

# ============================================================================
# ROUTES - BioKnowdes (AI Chat, Articles, Settings)
# ============================================================================

@app.route('/ask-lumi')
def chat():
    external_session_id = request.args.get('session_id')
    if external_session_id:
        session['session_id'] = external_session_id
    return render_template('chat.html')

@app.route('/ask-lumi/settings')
def settings():
    return render_template('settings.html')

@app.route('/ask-lumi/articles')
def articles_browser():
    """Article browser page"""
    return render_template('articles.html')

@app.route('/ask-lumi/external-demo')
def external_demo():
    """External integration demo page"""
    return send_file('external_button_example.html')

@app.route('/ask-lumi/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        text_content = extract_text_from_file(file_path, filename)
        
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        session_file = f'sessions/{session_id}.json'
        
        documents = []
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    documents = existing_data.get('documents', [])
            except:
                documents = []
        
        document_id = str(uuid.uuid4())
        new_document = {
            'id': document_id,
            'filename': filename,
            'content': text_content,
            'selected': True,
            'upload_time': str(uuid.uuid4())
        }
        
        documents.append(new_document)
        
        if len(documents) > 10:
            documents = documents[-10:]
        
        session_data = {
            'documents': documents
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'document_id': document_id,
            'content_preview': text_content[:500] + '...' if len(text_content) > 500 else text_content,
            'total_documents': len(documents)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask-lumi/chat', methods=['POST'])
def ask_lumi():
    """Endpoint for AI chat"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session found'}), 400
        
        session_file = f'sessions/{session_id}.json'
        if not os.path.exists(session_file):
            return jsonify({'error': 'No documents loaded'}), 400
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        documents = session_data.get('documents', [])
        selected_docs = [doc for doc in documents if doc.get('selected', False)]
        
        if not selected_docs:
            return jsonify({'error': 'No documents selected'}), 400
        
        context = "\n\n---\n\n".join([
            f"Document: {doc['filename']}\n\n{truncate_text_for_context(doc['content'])}"
            for doc in selected_docs
        ])
        
        settings_file = f'sessions/{session_id}_settings.json'
        settings = {}
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        
        provider = settings.get('provider', 'lm_studio')
        
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant analyzing scientific documents. Answer questions based on the following documents:\n\n{context}"
            },
            {
                "role": "user",
                "content": message
            }
        ]
        
        if provider == 'openai':
            ai_response = call_openai_api(messages, settings)
        elif provider == 'anthropic':
            ai_response = call_anthropic_api(messages, settings)
        elif provider == 'gemini':
            ai_response = call_gemini_api(messages, settings)
        else:
            ai_response = call_lm_studio_api(messages)
        
        return process_ai_response(ai_response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-lumi/documents')
def get_documents():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'documents': []})
    
    session_file = f'sessions/{session_id}.json'
    if not os.path.exists(session_file):
        return jsonify({'documents': []})
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        documents = session_data.get('documents', [])
        
        doc_list = []
        for doc in documents:
            content = doc.get('content', '')
            doc_info = {
                'id': doc['id'],
                'filename': doc['filename'],
                'selected': doc.get('selected', False),
                'preview': content[:200] + '...' if len(content) > 200 else content
            }
            
            if 'download_url' in doc:
                # Usar URL do S3 se houver pdf_filename
                if 'pdf_filename' in doc and doc['pdf_filename']:
                    doc_info['download_url'] = get_s3_pdf_url(doc['pdf_filename'])
                else:
                    doc_info['download_url'] = doc['download_url']
            if 'pmc_link' in doc:
                doc_info['pmc_link'] = doc['pmc_link']
            if 'pdf_filename' in doc:
                doc_info['pdf_filename'] = doc['pdf_filename']
                
            doc_list.append(doc_info)
        
        return jsonify({'documents': doc_list})
    except Exception as e:
        return jsonify({'error': 'Error reading documents'}), 500

@app.route('/ask-lumi/documents/toggle/<document_id>', methods=['POST'])
def toggle_document(document_id):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    session_file = f'sessions/{session_id}.json'
    if not os.path.exists(session_file):
        return jsonify({'error': 'Session not found'}), 400
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    
    documents = session_data.get('documents', [])
    
    for doc in documents:
        if doc['id'] == document_id:
            doc['selected'] = not doc.get('selected', False)
            break
    
    session_data['documents'] = documents
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False)
    
    return jsonify({'success': True})

@app.route('/ask-lumi/documents/delete/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    session_file = f'sessions/{session_id}.json'
    if not os.path.exists(session_file):
        return jsonify({'error': 'Session not found'}), 400
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    
    documents = session_data.get('documents', [])
    documents = [doc for doc in documents if doc['id'] != document_id]
    
    session_data['documents'] = documents
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False)
    
    return jsonify({'success': True})

@app.route('/ask-lumi/clear')
def clear_session():
    session_id = session.get('session_id')
    if session_id:
        session_file = f'sessions/{session_id}.json'
        if os.path.exists(session_file):
            os.remove(session_file)
    
    return jsonify({'success': True})

@app.route('/ask-lumi/api/load-articles', methods=['POST'])
def load_articles():
    """API endpoint to load articles by filename list"""
    try:
        data = request.get_json()
        if not data or 'filenames' not in data:
            return jsonify({'error': 'filenames list required'}), 400
        
        filenames = data['filenames']
        if not isinstance(filenames, list):
            return jsonify({'error': 'filenames must be a list'}), 400
        
        articles_map = load_articles_mapping()
        
        session_docs = []
        for filename in filenames:
            if not filename.endswith('_text.txt'):
                if filename.endswith('.txt'):
                    filename = filename.replace('.txt', '_text.txt')
                else:
                    filename = filename + '_text.txt'
            
            content = load_article_content(filename)
            if content:
                doc_id = str(uuid.uuid4())
                doc_info = {
                    'id': doc_id,
                    'filename': filename.replace('_text.txt', ''),
                    'content': content,
                    'preview': content[:200] + '...' if len(content) > 200 else content,
                    'selected': True,
                    'is_article': True
                }
                
                if filename in articles_map:
                    doc_info['pdf_filename'] = articles_map[filename]['pdf_filename']
                    # Usar URL do S3 para download
                    if articles_map[filename]['pdf_filename']:
                        doc_info['download_url'] = get_s3_pdf_url(articles_map[filename]['pdf_filename'])
                    else:
                        doc_info['download_url'] = articles_map[filename]['download_url']
                    doc_info['pmc_link'] = articles_map[filename]['pmc_link']
                    doc_info['title'] = articles_map[filename]['title']
                
                session_docs.append(doc_info)
        
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        os.makedirs('sessions', exist_ok=True)
        
        session_file = f'sessions/{session_id}.json'
        existing_docs = []
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    existing_docs = session_data.get('documents', [])
            except:
                existing_docs = []
        
        all_docs = existing_docs + session_docs
        
        session_data = {'documents': all_docs}
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'loaded_count': len(session_docs),
            'documents': session_docs,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-lumi/api/search-articles', methods=['POST'])
def search_articles_api():
    """API endpoint to search articles"""
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])
        
        results = search_articles_by_keywords(keywords)
        
        articles_map = load_articles_mapping()
        
        articles_data = []
        for filename in results[:50]:
            article_info = {
                'filename': filename,
                'display_name': filename.replace('_text.txt', '')
            }
            
            if filename in articles_map:
                article_info.update(articles_map[filename])
            
            articles_data.append(article_info)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'articles': articles_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-lumi/api/articles-list', methods=['GET'])
def articles_list():
    """API endpoint to get all available articles"""
    try:
        if not os.path.exists(ARTICLES_TXT_FOLDER):
            return jsonify({'success': False, 'error': 'Articles folder not found'}), 404
        
        articles_map = load_articles_mapping()
        
        articles = []
        for filename in os.listdir(ARTICLES_TXT_FOLDER):
            if filename.endswith('_text.txt'):
                article_info = {
                    'filename': filename,
                    'display_name': filename.replace('_text.txt', '')
                }
                
                if filename in articles_map:
                    article_info.update(articles_map[filename])
                    # Usar URL do S3 para download se houver pdf_filename
                    if 'pdf_filename' in articles_map[filename] and articles_map[filename]['pdf_filename']:
                        article_info['download_url'] = get_s3_pdf_url(articles_map[filename]['pdf_filename'])
                
                articles.append(article_info)
        
        return jsonify({
            'success': True,
            'count': len(articles),
            'articles': articles
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ask-lumi/api/settings', methods=['GET', 'POST'])
def api_settings():
    """API endpoint for settings"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    settings_file = f'sessions/{session_id}_settings.json'
    
    if request.method == 'GET':
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return jsonify(settings)
        else:
            return jsonify({'provider': 'lm_studio'})
    
    elif request.method == 'POST':
        settings = request.get_json()
        
        with open(settings_file, 'w') as f:
            json.dump(settings, f)
        
        return jsonify({'success': True})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    create_graph("")
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
