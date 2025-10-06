"""
NASA Research Assistant - Cleaned Version
A focused chatbot that retrieves relevant articles from keywords_resultados.jsonl
to help NASA researchers with contextual document access.
"""

from typing import List, Dict, Any, Tuple
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import numpy as np
import openai
import json
from flask import Flask, render_template_string, request, jsonify


# ==================== ARTICLE RANKER ====================

class ArticleRanker:
    """Loads and ranks articles from keywords_resultados.jsonl using semantic similarity."""
    
    def __init__(self, jsonl_path: str, openai_api_key: str = None):
        self.articles = []
        self.load_articles(jsonl_path)
        self.openai_api_key = openai_api_key
        self.embeddings_cache = {}
        self.cache_file = "article_embeddings_cache.json"
        
        # Load or create embeddings cache
        self.load_embeddings_cache()
        print(f"✓ OpenAI embeddings ready. {len(self.articles)} articles loaded with cache.")
        
    def load_articles(self, jsonl_path: str):
        """Load articles from JSONL file."""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                keywords = [kw[0] for kw in article['keywords']]
                self.articles.append({
                    'article': article['article'],
                    'keywords': keywords,
                    'keywords_with_scores': article['keywords']
                })
    
    def load_embeddings_cache(self):
        """Load embeddings cache from file or external URL."""
        # Try local cache first
        try:
            with open(self.cache_file, 'r') as f:
                self.embeddings_cache = json.load(f)
            print(f"✓ Loaded {len(self.embeddings_cache)} cached embeddings from local file")
            return
        except FileNotFoundError:
            pass
        
        # Try downloading from external URL (for Heroku)
        import os
        cache_url = os.getenv('EMBEDDINGS_CACHE_URL')
        if cache_url:
            print(f"📥 Downloading embeddings cache from {cache_url}...")
            try:
                import requests
                response = requests.get(cache_url, timeout=30)
                response.raise_for_status()
                self.embeddings_cache = response.json()
                print(f"✓ Loaded {len(self.embeddings_cache)} cached embeddings from external URL")
                return
            except Exception as e:
                print(f"❌ Failed to download cache: {e}")
        
        # Fallback: create new cache if possible
        print("📦 No cache found. Checking if we can create one...")
        self.embeddings_cache = {}
        
        # Check if we're in a read-only environment (like Heroku)
        if self.openai_api_key and self._can_write_cache():
            print("🔄 Precomputing embeddings cache...")
            self.precompute_article_embeddings()
        else:
            print("⚠️ Read-only environment detected. Using on-demand embeddings.")
    
    def _can_write_cache(self):
        """Check if we can write to the filesystem."""
        try:
            test_file = f"{self.cache_file}.test"
            with open(test_file, 'w') as f:
                f.write('test')
            import os
            os.remove(test_file)
            return True
        except (OSError, PermissionError):
            return False
    
    def save_embeddings_cache(self):
        """Save embeddings cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.embeddings_cache, f)
            print(f"💾 Saved {len(self.embeddings_cache)} embeddings to cache")
        except (OSError, PermissionError):
            print("⚠️ Cannot save cache in read-only environment")
    
    def precompute_article_embeddings(self):
        """Precompute embeddings for all articles."""
        print(f"🔄 Precomputing embeddings for {len(self.articles)} articles...")
        
        for idx, article in enumerate(self.articles):
            article_text = " ".join(article['keywords'])
            cache_key = f"article_{idx}"
            
            if cache_key not in self.embeddings_cache:
                embedding = self.get_openai_embedding(article_text)
                self.embeddings_cache[cache_key] = embedding
                
                # Save cache every 10 articles to avoid losing progress
                if (idx + 1) % 10 == 0:
                    self.save_embeddings_cache()
                    print(f"  Progress: {idx + 1}/{len(self.articles)} articles")
        
        # Final save
        self.save_embeddings_cache()
        print("✅ All article embeddings precomputed!")
    
    def get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API."""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            # Fallback: return zero vector
            return [0.0] * 1536
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rank_by_embeddings(self, query_keywords: List[str], top_n: int = 5) -> List[Dict]:
        """
        Rank articles using cached OpenAI embeddings, with fallback for read-only environments.
        
        Returns:
            List of dicts with article metadata and relevance scores
        """
        query_text = " ".join(query_keywords)
        
        # Cache query embeddings if possible
        query_cache_key = f"query_{hash(query_text)}"
        if query_cache_key in self.embeddings_cache:
            query_embedding = self.embeddings_cache[query_cache_key]
        else:
            query_embedding = self.get_openai_embedding(query_text)
            if self._can_write_cache():
                self.embeddings_cache[query_cache_key] = query_embedding
        
        # If we don't have cached embeddings and can't write cache (Heroku), 
        # use keyword matching for pre-filtering to reduce API calls
        if not self.embeddings_cache or len([k for k in self.embeddings_cache.keys() if k.startswith('article_')]) < len(self.articles) // 2:
            # Pre-filter articles using keyword matching to reduce embeddings needed
            candidate_articles = self._prefilter_articles_by_keywords(query_keywords, top_n * 3)
        else:
            # Use all articles if we have good cache coverage
            candidate_articles = [(idx, article) for idx, article in enumerate(self.articles)]
        
        scores = []
        for idx, article in candidate_articles:
            # Use cached article embedding if available
            cache_key = f"article_{idx}"
            if cache_key in self.embeddings_cache:
                article_embedding = self.embeddings_cache[cache_key]
            else:
                # Compute embedding on-demand
                article_text = " ".join(article['keywords'])
                article_embedding = self.get_openai_embedding(article_text)
                if self._can_write_cache():
                    self.embeddings_cache[cache_key] = article_embedding
            
            similarity = self.cosine_similarity(query_embedding, article_embedding)
            scores.append({
                'id': idx,
                'article': article['article'],
                'score': float(similarity),
                'keywords': article['keywords'][:10],
                'keywords_with_scores': article['keywords_with_scores'][:10]
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_n]
    
    def _prefilter_articles_by_keywords(self, query_keywords: List[str], limit: int = 15) -> List[tuple]:
        """Pre-filter articles by keyword overlap to reduce embedding API calls."""
        query_words = set(word.lower() for word in query_keywords)
        
        scores = []
        for idx, article in enumerate(self.articles):
            article_words = set(word.lower() for word in article['keywords'])
            overlap = len(query_words.intersection(article_words))
            if overlap > 0:
                scores.append((overlap, idx, article))
        
        # Sort by keyword overlap and return top candidates
        scores.sort(reverse=True, key=lambda x: x[0])
        return [(idx, article) for _, idx, article in scores[:limit]]
    
    def get_article_by_id(self, article_id: int) -> Dict:
        """Retrieve full article metadata by ID."""
        if 0 <= article_id < len(self.articles):
            return self.articles[article_id]
        return None


# ==================== CUSTOM TOOL ====================

class ArticleSearchInput(BaseModel):
    """Input schema for article search."""
    query_keywords: List[str] = Field(description="Keywords to search for relevant NASA research articles")
    top_n: int = Field(default=5, description="Number of top articles to return")


class NASAArticleSearchTool(BaseTool):
    """Tool to search NASA research articles from local database."""
    
    name: str = "nasa_article_search"
    description: str = """
    Search for relevant NASA research articles from the curated database.
    Use this when researchers ask questions about space, aeronautics, or scientific research.
    
    Input: List of keywords related to the research topic
    Output: Top ranked articles with relevance scores and keywords
    """
    args_schema: type[BaseModel] = ArticleSearchInput
    ranker: Any = Field(default=None)
    
    def __init__(self, ranker):
        super().__init__(ranker=ranker)
    
    def _run(self, query_keywords: List[str], top_n: int = 5) -> str:
        """Execute article search and return formatted results."""
        results = self.ranker.rank_by_embeddings(query_keywords, top_n)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result['id'],
                "article": result['article'],
                "relevance_score": round(result['score'], 4),
                "keywords": result['keywords'],
                "confidence": self._classify_score(result['score'])
            })
        
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
    
    def _classify_score(self, score: float) -> str:
        """Classify relevance strength."""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        elif score >= 0.3:
            return "LOW"
        return "VERY_LOW"


# ==================== RESEARCH AGENT ====================

class NASAResearchAgent:
    """NASA Research Assistant Agent with conversational memory."""
    
    def __init__(
        self, 
        article_ranker: ArticleRanker, 
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ):
        self.ranker = article_ranker
        self.last_retrieved_articles = []  # Store last search results
        
        # Configure LLM
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,
            base_url=base_url,
        )
        
        # Configure memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Configure tools
        self.tools = [NASAArticleSearchTool(ranker=self.ranker)]
        
        # Configure prompt
        self.prompt = self._create_prompt()
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create system prompt for NASA research assistant."""
        system_message = """You are a NASA Research Assistant specialized in helping researchers find and understand relevant scientific articles.

YOUR ROLE:
- Help NASA researchers find relevant articles from the curated database
- Provide clear, concise summaries of research findings
- Extract key information and keywords from user queries
- Cite specific articles with their relevance scores

TOOL USAGE:
- Use 'nasa_article_search' to find relevant articles
- Extract meaningful keywords from user questions (focus on technical terms, concepts, topics)
- Always provide article IDs and relevance scores in your responses

RESPONSE FORMAT:
1. **SUMÁRIO DO TEMA**: Start by providing a comprehensive overview of what the NASA research articles say about the topic the user is asking about, highlighting key findings, research areas, and scientific insights from the database
2. **BUSCA POR ARTIGOS**: Search for relevant articles using extracted keywords
3. **RECOMENDAÇÕES**: Present the most relevant articles with specific references and relevance scores
4. **ANÁLISE**: Explain how the recommended articles support and expand on the topic summary
5. **PRÓXIMOS PASSOS**: Suggest follow-up questions or additional research directions if appropriate

IMPORTANT:
- Always start with a comprehensive summary of what the NASA research articles say about the user's topic
- This summary should synthesize the collective knowledge from the articles, not just restate the user's question
- Cite article names and their relevance scores
- Be specific about which articles support your answers
- If relevance is LOW or VERY_LOW, mention this uncertainty
- Keep responses professional and research-focused
- The topic summary should give users a broad understanding of the research landscape on their subject"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def research(self, question: str) -> Dict[str, Any]:
        """Execute research query and return results with retrieved articles."""
        result = self.agent_executor.invoke({"input": question})
        
        # Extract retrieved article IDs from intermediate steps
        retrieved_ids = []
        if result.get("intermediate_steps"):
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if action.tool == "nasa_article_search":
                        try:
                            articles_data = json.loads(observation)
                            retrieved_ids = [article['id'] for article in articles_data]
                            self.last_retrieved_articles = articles_data
                        except:
                            pass
        
        return {
            "answer": result["output"],
            "retrieved_article_ids": retrieved_ids,
            "retrieved_articles": self.last_retrieved_articles
        }
    
    def get_article_details(self, article_id: int) -> Dict:
        """Get full details of a specific article."""
        return self.ranker.get_article_by_id(article_id)
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


# ==================== FLASK APPLICATION ====================

app = Flask(__name__)

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the NASA research agent."""
    global agent
    print("Initializing NASA Research Assistant...")
    ranker = ArticleRanker("keywords_resultados.jsonl")
    agent = NASAResearchAgent(
        article_ranker=ranker,
        model="llama3.1:8b",
        base_url="http://localhost:11434"
    )
    print("✓ Agent ready!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NASA Research Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f1419;
            color: #e8e8e8;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: #1a1f2e;
            padding: 20px;
            border-bottom: 2px solid #2a3f5f;
            text-align: center;
        }
        
        .header h1 {
            color: #4a9eff;
            font-size: 24px;
        }
        
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        .sidebar {
            width: 350px;
            background: #1a1f2e;
            border-right: 2px solid #2a3f5f;
            overflow-y: auto;
            padding: 20px;
        }
        
        .sidebar h2 {
            color: #4a9eff;
            font-size: 18px;
            margin-bottom: 15px;
        }
        
        .article-item {
            background: #252d3d;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .article-item:hover {
            background: #2d3548;
            transform: translateX(5px);
        }
        
        .article-name {
            color: #4a9eff;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .article-score {
            color: #888;
            font-size: 12px;
        }
        
        .score-high { color: #4caf50; }
        .score-medium { color: #ff9800; }
        .score-low { color: #f44336; }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0f1419;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        
        .message.user {
            background: #2a3f5f;
            margin-left: auto;
        }
        
        .message.assistant {
            background: #1a2332;
        }
        
        .message-label {
            font-weight: 600;
            color: #4a9eff;
            margin-bottom: 8px;
        }
        
        .input-container {
            padding: 20px;
            background: #1a1f2e;
            border-top: 2px solid #2a3f5f;
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 12px;
            background: #252d3d;
            border: 1px solid #2a3f5f;
            border-radius: 6px;
            color: #e8e8e8;
            font-size: 14px;
        }
        
        .input-container button {
            padding: 12px 30px;
            background: #4a9eff;
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .input-container button:hover {
            background: #357abd;
        }
        
        .input-container button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        
        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 8px;
        }
        
        .keyword {
            background: #2a3f5f;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 NASA Research Assistant</h1>
    </div>
    
    <div class="main-container">
        <div class="sidebar">
            <h2>📚 Retrieved Documents</h2>
            <div id="articles-list">
                <p style="color: #666;">Articles will appear here after your first query...</p>
            </div>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages"></div>
            
            <div class="input-container">
                <input type="text" id="user-input" placeholder="Ask about NASA research..." />
                <button id="send-btn">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        const messagesDiv = document.getElementById('messages');
        const articlesDiv = document.getElementById('articles-list');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        
        function addMessage(text, isUser) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            msgDiv.innerHTML = `
                <div class="message-label">${isUser ? 'You' : 'NASA Assistant'}</div>
                <div>${text}</div>
            `;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function updateArticles(articles) {
            if (!articles || articles.length === 0) {
                articlesDiv.innerHTML = '<p style="color: #666;">No articles found for this query.</p>';
                return;
            }
            
            articlesDiv.innerHTML = '';
            articles.forEach(article => {
                const scoreClass = article.confidence === 'HIGH' ? 'score-high' : 
                                   article.confidence === 'MEDIUM' ? 'score-medium' : 'score-low';
                
                const keywords = article.keywords.slice(0, 5).map(kw => 
                    `<span class="keyword">${kw}</span>`
                ).join('');
                
                const div = document.createElement('div');
                div.className = 'article-item';
                div.innerHTML = `
                    <div class="article-name">${article.article}</div>
                    <div class="article-score ${scoreClass}">
                        Score: ${article.relevance_score} (${article.confidence})
                    </div>
                    <div class="keywords">${keywords}</div>
                `;
                articlesDiv.appendChild(div);
            });
        }
        
        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;
            
            addMessage(question, true);
            userInput.value = '';
            sendBtn.disabled = true;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                
                const data = await response.json();
                addMessage(data.answer, false);
                updateArticles(data.retrieved_articles);
            } catch (error) {
                addMessage('Error: Could not connect to the server.', false);
            } finally {
                sendBtn.disabled = false;
                userInput.focus();
            }
        }
        
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render main interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        result = agent.research(question)
        return jsonify({
            'answer': result['answer'],
            'retrieved_articles': result['retrieved_articles']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation history."""
    agent.clear_memory()
    return jsonify({'status': 'success'})


# ==================== MAIN ====================

if __name__ == "__main__":
    initialize_agent()
    print("\n" + "="*60)
    print("🚀 NASA Research Assistant is running!")
    print("="*60)
    print("\n📍 Open your browser and go to: http://localhost:5000")
    print("\n")
    app.run(debug=True, host='0.0.0.0', port=5000)