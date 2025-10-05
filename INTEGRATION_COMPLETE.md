# ✅ Integração Completa - BioKnowdes

## 🎉 O que foi integrado

Seu projeto BioKnowdes foi **totalmente integrado** ao projeto bioknow-nasa!

## 📁 Estrutura Final

```
bioknow-nasa/
├── app.py                          ✅ INTEGRADO (ambos os projetos mesclados)
├── requirements.txt                ✅ ATUALIZADO (todas as dependências)
├── templates/
│   ├── index.html                 ✅ ORIGINAL (home com grafo)
│   ├── graph.html                 ✅ ORIGINAL (busca detalhada)
│   ├── heatmap.html               ✅ ORIGINAL (análise térmica)
│   ├── chat.html                  ✅ NOVO (seu chat com IA)
│   ├── articles.html              ✅ NOVO (navegador de artigos)
│   └── settings.html              ✅ NOVO (configurações de IA)
├── static/
│   ├── css/
│   │   ├── style.css             ✅ ORIGINAL (estilos do grafo)
│   │   ├── bioknowdes.css        ✅ NOVO (seus estilos)
│   │   ├── articles.css          ✅ NOVO
│   │   └── settings.css          ✅ NOVO
│   ├── js/
│   │   ├── script.js             ✅ ORIGINAL (lógica do grafo)
│   │   ├── bioknowdes.js         ✅ NOVO (sua lógica)
│   │   ├── articles.js           ✅ NOVO
│   │   └── settings.js           ✅ NOVO
│   └── images/                   ✅ MANTIDO
├── data/                          ✅ COPIADO (artigos NASA)
├── sessions/                      ✅ CRIADO (para sessões)
├── uploads/                       ✅ CRIADO (para uploads)
└── external_button_example.html   ✅ COPIADO
```

## 🗺️ Mapa de Rotas

| Rota | Funcionalidade | Origem |
|------|---------------|--------|
| `/` | Home (visualização de grafo) | Original |
| `/graph` | Busca detalhada | Original |
| `/heatmap` | Análise de similaridade | Original |
| `/chat` | Chat com IA | **BioKnowdes** |
| `/articles` | Navegador de artigos NASA | **BioKnowdes** |
| `/settings` | Configurações de IA | **BioKnowdes** |
| `/ask-lumi` | API do chat (endpoint principal) | **BioKnowdes** |
| `/api/load-articles` | Carregar artigos | **BioKnowdes** |
| `/api/articles-list` | Listar todos os artigos | **BioKnowdes** |
| `/api/search-articles` | Buscar artigos | **BioKnowdes** |
| `/api/settings` | Salvar/carregar configurações | **BioKnowdes** |
| `/documents` | Listar documentos da sessão | **BioKnowdes** |
| `/upload` | Upload de documentos | **BioKnowdes** |
| `/external-demo` | Exemplo de integração externa | **BioKnowdes** |

## 🚀 Como Executar

```bash
# 1. Entre no diretório
cd /home/jabs/development/personal/Nasa-Space-Apps_25/bioknow-nasa

# 2. Ative o ambiente virtual (ou crie um novo)
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o servidor
python app.py
```

## 🌐 Acessando o Sistema

Após iniciar o servidor, acesse:

**http://localhost:5000**

### Navegação:

1. **Home (`/`)**: 
   - Visualização do grafo de keywords
   - Menu dropdown com todas as opções

2. **AI Chat (`/chat`)**:
   - Carregue artigos
   - Faça perguntas sobre os documentos
   - Configure qual IA usar (OpenAI, Anthropic, Gemini, LM Studio)

3. **Browse Articles (`/articles`)**:
   - Explore +600 artigos NASA
   - Busque por keywords
   - Carregue diretamente no chat

4. **Heatmap (`/heatmap`)**:
   - Análise de similaridade entre artigos

5. **Settings (`/settings`)**:
   - Configure API keys
   - Escolha o provider de IA

## 🔑 Configurações de IA

O sistema suporta 4 providers:

### 1. LM Studio (Local)
- Padrão, não precisa de API key
- Rodando em: `http://{windows_ip}:3000`

### 2. OpenAI
- Configure em `/settings`
- Precisa de: `OPENAI_API_KEY`

### 3. Anthropic (Claude)
- Configure em `/settings`
- Precisa de: `ANTHROPIC_API_KEY`

### 4. Google Gemini
- Configure em `/settings`
- Precisa de: `GEMINI_API_KEY`

## ✨ Funcionalidades Integradas

### Do Projeto Original:
- ✅ Visualização de grafo de keywords
- ✅ Filtros e busca no grafo
- ✅ Heatmap de similaridade
- ✅ Interface dark theme

### Do BioKnowdes:
- ✅ Chat com IA sobre documentos
- ✅ Navegador de +600 artigos NASA
- ✅ Upload de PDFs/DOCXs
- ✅ Suporte a múltiplas IAs
- ✅ Sistema de sessões
- ✅ Seleção dinâmica de documentos
- ✅ Botões Full Text (PMC) e Download
- ✅ API externa para integração

## 🧪 Testando

### 1. Teste o Grafo Original
```
http://localhost:5000/
http://localhost:5000/graph
http://localhost:5000/heatmap
```

### 2. Teste o Chat com IA
```
1. Vá para http://localhost:5000/chat
2. Clique em "Browse Articles"
3. Busque por keywords (ex: "microgravity")
4. Clique em "Load Selected"
5. Volte ao chat e faça uma pergunta!
```

### 3. Teste a Navegação
- Use o menu dropdown na home
- Navegue entre todas as páginas
- Verifique se os links funcionam

## 📝 Notas Importantes

### CORS Habilitado
As rotas `/api/*` aceitam requisições externas para integração

### Sessões
- Cada usuário tem uma sessão única
- Documentos são salvos em `sessions/`
- Configurações salvas separadamente

### Dados
- Artigos NASA em `data/processed/`
- Mapeamento em `data/articles_mapping.tsv`

## 🐛 Solução de Problemas

### Erro: "No module named 'flask_cors'"
```bash
pip install flask-cors
```

### Erro: "Grafo não encontrado"
```bash
# Certifique-se de que existe:
ls grafo_keywords.gpickle
```

### Erro: "Articles folder not found"
```bash
# Verifique se a pasta data foi copiada:
ls data/processed/
```

### Chat não responde
- Verifique se carregou documentos
- Verifique se selecionou os documentos (olho verde)
- Verifique se a IA está configurada

## 🎯 Endpoints da API

### Para Integração Externa:

```javascript
// Carregar artigos
POST /api/load-articles
{
  "filenames": ["article1_text.txt", "article2_text.txt"]
}

// Listar todos os artigos
GET /api/articles-list

// Fazer pergunta ao chat
POST /ask-lumi
{
  "message": "What are the effects of microgravity?"
}
```

## 🎊 Sucesso!

Você agora tem um sistema completo que combina:
- 📊 Visualização de grafos e análise de keywords
- 💬 Chat inteligente com IA
- 📚 Navegação de artigos científicos NASA
- 🔧 Configuração flexível de providers de IA
- 🔗 API para integração externa

**Bom trabalho! 🚀**

