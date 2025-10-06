# 🔄 Route Update - BioKnowdes

## 🔧 Environment Variable Configuration

### 📋 Initial Setup

1. **Copy the environment variable template:**
   ```bash
   cp env.template .env
   ```

2. **Set your API keys in the file `.env`:**
   ```bash
   # Edit the .env file with your real keys
   nano .env
   ```

3. **Mandatory variables for production:**
   ```env
   # OpenAI API (mandatory for full functionality)
   OPENAI_API_KEY=sk-your-openai-key-here
   
   # Flask secret key (mandatory for sessions)
   FLASK_SECRET_KEY=your-secret-key-here
   
   # Other APIs (Optional)
   ANTHROPIC_API_KEY=your-anthropic-key-here
   GEMINI_API_KEY=your-gemini-key-here
   ```

### 🔒 Production Security

- ✅ **Never commit the file `.env`** (its already on `.gitignore`)
- ✅ **Use different keys** for development and production
- ✅ **Rotate your keys** periodically
- ✅ **Set environment variables** on your hosting provider

### 🚀 Production Deployment

For platforms such as Heroku, Railway, or Vercel:
```bash
#Configure the environment variables on the provider panel
OPENAI_API_KEY=sk-your-production-key
FLASK_SECRET_KEY=your-production-secret-key
FLASK_ENV=production
```

## ✅ Applied Changes

All of the BioKnowdes routes are now under the prefix `/ask-lumi`: 

### 📋 Route Mapping

| Old Route | New Route | Description |
|-------------|-----------|-----------|
| `/chat` | `/ask-lumi` | Main chat page |
| `/articles` | `/ask-lumi/articles` | Article browser |
| `/settings` | `/ask-lumi/settings` | AI settings |
| `/upload` | `/ask-lumi/upload` | Document upload |
| `/documents` | `/ask-lumi/documents` | List documents |
| `/documents/toggle/:id` | `/ask-lumi/documents/toggle/:id` | Toggle selection |
| `/documents/delete/:id` | `/ask-lumi/documents/delete/:id` | Delete document |
| `/clear` | `/ask-lumi/clear` | Clear session |
| `/external-demo` | `/ask-lumi/external-demo` | External example |

### 🔌 API Endpoints

| Old Route | New Route | Method | Description |
|-------------|-----------|--------|-----------|
| `/chat` (POST) | `/ask-lumi/chat` | POST | AI Chat |
| `/api/load-articles` | `/ask-lumi/api/load-articles` | POST | Load articles |
| `/api/articles-list` | `/ask-lumi/api/articles-list` | GET | List articles |
| `/api/search-articles` | `/ask-lumi/api/search-articles` | POST | Search articles |
| `/api/settings` | `/ask-lumi/api/settings` | GET/POST | AI Config |

### 🌐 Original Routes (Kept)

These are the original project routes that **were not changed**:

- `/` - Home with graph
- `/graph` - Graph search
- `/heatmap` - Thermal analysis
- `/get_graph_data` - Graph data
- `/api/keywords` - Keywords list

## 📁 Updated Files

### Backend:
- ✅ `app.py` - All of the BioKnowdes routes altered
- ✅ CORS updated to `/ask-lumi/api/*`

### Templates:
- ✅ `templates/index.html` - Menu links 
- ✅ `templates/chat.html` - Navigation links
- ✅ `templates/articles.html` - Link "Back to Chat"
- ✅ `templates/settings.html` - Link "Back to Chat"

### JavaScript:
- ✅ `static/js/bioknowdes.js` - All fetch calls
- ✅ `static/js/articles.js` - API calls and redirections
- ✅ `static/js/settings.js` - API calls and redirections

### Outros:
- ✅ `external_button_example.html` - API URL and redirect

## 🚀 How to Access

### Main Pages:

```
Home (Graph):        http://localhost:5000/
AI Chat:             http://localhost:5000/ask-lumi
Browse Articles:     http://localhost:5000/ask-lumi/articles
Settings:            http://localhost:5000/ask-lumi/settings
Heatmap:             http://localhost:5000/heatmap
```

### API Endpoints:

```bash
# AI Chat
POST http://localhost:5000/ask-lumi/chat
{
  "message": "Your question here"
}

# Load articles
POST http://localhost:5000/ask-lumi/api/load-articles
{
  "filenames": ["article1_text.txt", "article2_text.txt"]
}

# List all articles
GET http://localhost:5000/ask-lumi/api/articles-list

# Settings
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

## 🧪 Testing

Execute the server:

```bash
cd /home/jabs/development/personal/Nasa-Space-Apps_25/bioknow-nasa
source venv/bin/activate
python app.py
```

Test each route:

```bash
# Original home
curl http://localhost:5000/

# Chat (must return HTML)
curl http://localhost:5000/ask-lumi

# Articles API
curl http://localhost:5000/ask-lumi/api/articles-list
```

## 📊 Navigation Structure

```
┌─────────────┐
│    Home     │ (Graph - /)
│   (Graph)   │
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

## 🔄 Workflow

1. **User acess home** (`/`)
   - Sees graph visualization
   - Dropdown menu with all options

2. **Clicks "AI Chat"** → Goes to `/ask-lumi`
   - Can load documents
   - Can go to `/ask-lumi/articles` or `/ask-lumi/settings`

3. **Loads articles on** `/ask-lumi/articles`
   - Selects articles
   - Clicks "Load Selected"
   - API: `POST /ask-lumi/api/load-articles`
   - Redirects to `/ask-lumi` with session_id

4. **Asks questions on the chat**
   - API: `POST /ask-lumi/chat`
   - AI answers based on documents

## 🎯 Benefits of the New Structure

✅ **Organization**: All BioKnowdes routes grouped under `/ask-lumi` 

✅ **Separation**: Clear distinction between:
   - Graph visualization (original)
   - AI and document analysis (BioKnowdes)

✅ **Scalability**: Each to add new features under `/ask-lumi`

✅ **External API**: CORS configured to `/ask-lumi/api/*`

## 📝 Important Notes

1. **Session ID**: Always included in API responses

2. **CORS**: Allowed only for `/ask-lumi/api/*`, for security 

3. **Navigation**: All templates have updated links

4. **redirects**: After loading articles or saving settings, redirects to `/ask-lumi`

## 🐛 Troubleshooting

### 404 Error in routes
```bash
# Verify if its accessing with the correct prefix
# WRONG:  /articles
# CORRECT: /ask-lumi/articles
```

### CORS Error
```bash
# Make sure its requisition goes to /ask-lumi/api/*
# CORS is configured only for this prefix
```

### Session not persisting
```bash
# Verify if the session_id is being passed correctly
# On the URL after loading articles
```

---

**Full Update! 🎉**

All routes are now organized under `/ask-lumi`. 

