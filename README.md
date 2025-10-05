# 🔄 Atualização de Rotas - BioKnowdes

## ✅ Mudanças Aplicadas

Todas as rotas do BioKnowdes agora estão sob o prefixo `/ask-lumi`:

### 📋 Mapeamento de Rotas

| Rota Antiga | Rota Nova | Descrição |
|-------------|-----------|-----------|
| `/chat` | `/ask-lumi` | Página principal do chat |
| `/articles` | `/ask-lumi/articles` | Navegador de artigos |
| `/settings` | `/ask-lumi/settings` | Configurações de IA |
| `/upload` | `/ask-lumi/upload` | Upload de documentos |
| `/documents` | `/ask-lumi/documents` | Listar documentos |
| `/documents/toggle/:id` | `/ask-lumi/documents/toggle/:id` | Toggle seleção |
| `/documents/delete/:id` | `/ask-lumi/documents/delete/:id` | Deletar documento |
| `/clear` | `/ask-lumi/clear` | Limpar sessão |
| `/external-demo` | `/ask-lumi/external-demo` | Exemplo externo |

### 🔌 API Endpoints

| Rota Antiga | Rota Nova | Método | Descrição |
|-------------|-----------|--------|-----------|
| `/chat` (POST) | `/ask-lumi/chat` | POST | Chat com IA |
| `/api/load-articles` | `/ask-lumi/api/load-articles` | POST | Carregar artigos |
| `/api/articles-list` | `/ask-lumi/api/articles-list` | GET | Listar artigos |
| `/api/search-articles` | `/ask-lumi/api/search-articles` | POST | Buscar artigos |
| `/api/settings` | `/ask-lumi/api/settings` | GET/POST | Config da IA |

### 🌐 Rotas Originais (Mantidas)

Estas rotas do projeto original **não foram alteradas**:

- `/` - Home com grafo
- `/graph` - Busca no grafo
- `/heatmap` - Análise térmica
- `/get_graph_data` - Dados do grafo
- `/api/keywords` - Lista de keywords

## 📁 Arquivos Atualizados

### Backend:
- ✅ `app.py` - Todas as rotas do BioKnowdes alteradas
- ✅ CORS atualizado para `/ask-lumi/api/*`

### Templates:
- ✅ `templates/index.html` - Links do menu
- ✅ `templates/chat.html` - Links de navegação
- ✅ `templates/articles.html` - Link "Back to Chat"
- ✅ `templates/settings.html` - Link "Back to Chat"

### JavaScript:
- ✅ `static/js/bioknowdes.js` - Todas as chamadas fetch
- ✅ `static/js/articles.js` - API calls e redirecionamentos
- ✅ `static/js/settings.js` - API calls e redirecionamentos

### Outros:
- ✅ `external_button_example.html` - URL da API e redirect

## 🚀 Como Acessar

### Páginas Principais:

```
Home (Grafo):        http://localhost:5000/
AI Chat:             http://localhost:5000/ask-lumi
Browse Articles:     http://localhost:5000/ask-lumi/articles
Settings:            http://localhost:5000/ask-lumi/settings
Heatmap:             http://localhost:5000/heatmap
```

### API Endpoints:

```bash
# Chat com IA
POST http://localhost:5000/ask-lumi/chat
{
  "message": "Your question here"
}

# Carregar artigos
POST http://localhost:5000/ask-lumi/api/load-articles
{
  "filenames": ["article1_text.txt", "article2_text.txt"]
}

# Listar todos os artigos
GET http://localhost:5000/ask-lumi/api/articles-list

# Configurações
GET http://localhost:5000/ask-lumi/api/settings
POST http://localhost:5000/ask-lumi/api/settings
{
  "provider": "openai",
  "openai": {
    "api_key": "...",
    "model": "gpt-4"
  }
}
```

## 🧪 Testando

Execute o servidor:

```bash
cd /home/jabs/development/personal/Nasa-Space-Apps_25/bioknow-nasa
source venv/bin/activate
python app.py
```

Teste cada rota:

```bash
# Home original
curl http://localhost:5000/

# Chat (deve retornar HTML)
curl http://localhost:5000/ask-lumi

# API de artigos
curl http://localhost:5000/ask-lumi/api/articles-list
```

## 📊 Estrutura de Navegação

```
┌─────────────┐
│    Home     │ (Grafo - /)
│   (Grafo)   │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
┌──────▼──────┐  ┌──▼────────┐
│   Heatmap   │  │  Ask-Lumi │ (/ask-lumi)
│ (/heatmap)  │  │  (Chat)   │
└─────────────┘  └─────┬─────┘
                       │
                       ├────────────┬──────────────┐
                       │            │              │
                ┌──────▼──────┐ ┌──▼────────┐ ┌──▼─────────┐
                │   Articles  │ │ Settings  │ │  External  │
                │  (/articles)│ │(/settings)│ │   Demo     │
                └─────────────┘ └───────────┘ └────────────┘
```

## 🔄 Fluxo de Trabalho

1. **Usuário acessa home** (`/`)
   - Vê visualização do grafo
   - Menu dropdown com todas as opções

2. **Clica em "AI Chat"** → Vai para `/ask-lumi`
   - Pode carregar documentos
   - Pode ir para `/ask-lumi/articles` ou `/ask-lumi/settings`

3. **Carrega artigos em** `/ask-lumi/articles`
   - Seleciona artigos
   - Clica "Load Selected"
   - API: `POST /ask-lumi/api/load-articles`
   - Redireciona para `/ask-lumi` com session_id

4. **Faz perguntas no chat**
   - API: `POST /ask-lumi/chat`
   - IA responde baseada nos documentos

## 🎯 Benefícios da Nova Estrutura

✅ **Organização**: Todas as rotas do BioKnowdes agrupadas sob `/ask-lumi`

✅ **Separação**: Clara distinção entre:
   - Visualização de grafos (original)
   - IA e análise de documentos (BioKnowdes)

✅ **Escalabilidade**: Fácil adicionar novas features sob `/ask-lumi`

✅ **API Externa**: CORS configurado para `/ask-lumi/api/*`

## 📝 Notas Importantes

1. **Session ID**: Sempre incluído nas respostas da API para manter estado

2. **CORS**: Habilitado apenas para `/ask-lumi/api/*` para segurança

3. **Navegação**: Todos os templates têm links atualizados

4. **Redirecionamentos**: Após carregar artigos ou salvar settings, redireciona para `/ask-lumi`

## 🐛 Troubleshooting

### Erro 404 nas rotas
```bash
# Verifique se está acessando com o prefixo correto
# ERRADO:  /articles
# CORRETO: /ask-lumi/articles
```

### CORS Error
```bash
# Certifique-se que está fazendo requisição para /ask-lumi/api/*
# O CORS está configurado apenas para este prefixo
```

### Session não persiste
```bash
# Verifique se o session_id está sendo passado corretamente
# na URL após carregar artigos
```

---

**Atualização completa! 🎉**

Todas as rotas do BioKnowdes agora estão organizadas sob `/ask-lumi`.

