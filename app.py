from flask import Flask, render_template, request, jsonify, session, send_file, redirect
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
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# Import unified session manager (after load_dotenv)
from session_manager import get_session_manager

# Initialize session manager after environment variables are loaded
session_manager = get_session_manager()

# AmandaChatbot imports
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import numpy as np


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
# Use environment variable for secret key in production, fallback to fixed key for development
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'bioknow-dev-secret-key-2024')

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
# AMANDACHATBOT CLASSES
# ============================================================================

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
        cache_url = os.getenv('EMBEDDINGS_CACHE_URL', 'https://nasa-spaceapps-25.s3.us-east-1.amazonaws.com/article_embeddings_cache.json')
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
            print(f"❌ OpenAI embedding error: {e}")
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


class HomePageArticleShowInput(BaseModel):
    """Input schema for showing articles on homepage."""
    query_keywords: List[str] = Field(description="Keywords to search for relevant NASA research articles")
    top_n: int = Field(default=5, description="Number of top articles to return")
    

class HomePageArticleShowTool(BaseTool):
    """Tool to find and present NASA articles for homepage chat with user-friendly response."""
    
    name: str = "homepage_show_articles"
    description: str = """
    Use this tool when you want to show specific NASA research articles to the user on the homepage chat.
    This should be used when:
    1. The user has asked a specific research question
    2. You believe there are relevant articles that would help them
    3. You want to provide concrete research articles instead of just general guidance
    
    This tool will search for articles and return them in a format ready for the homepage chat.
    Use this when the conversation has progressed enough that showing specific articles would be valuable.
    """
    args_schema: type[BaseModel] = HomePageArticleShowInput
    ranker: Any = Field(default=None)
    
    def __init__(self, ranker):
        super().__init__(ranker=ranker)
    
    def _run(self, query_keywords: List[str], top_n: int = 5) -> str:
        """Search for articles and return them formatted for homepage display."""
        results = self.ranker.rank_by_embeddings(query_keywords, top_n)
        
        # Filter for good quality articles
        good_articles = [r for r in results if r.get('score', 0) > 0.25]
        
        if not good_articles:
            return "NO_ARTICLES_FOUND"
        
        # Ensure we have exactly 5 articles
        if len(good_articles) < 5:
            good_articles = results[:5]  # Take top 5 if not enough good ones
        else:
            good_articles = good_articles[:5]  # Limit to 5
        
        formatted_results = []
        for result in good_articles:
            formatted_results.append({
                "article": result['article'],
                "score": round(result.get('score', 0), 4),
                "keywords": result.get('keywords', [])[:5]
            })
        
        return json.dumps({
            "articles_found": len(formatted_results),
            "articles": formatted_results,
            "should_show_button": True
        }, ensure_ascii=False)


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
        system_message = """You are Lumi, a NASA Research Assistant specialized in helping researchers find and understand relevant scientific articles.

YOUR ROLE:
- Help researchers find the most relevant NASA articles from the curated database
- Provide clear, informative summaries of research findings
- Recommend specific articles that best address the user's question
- Always mention which articles are most useful for deeper study

TOOL USAGE:
- Use 'nasa_article_search' to find relevant articles for every question
- Extract meaningful keywords from user questions (focus on scientific terms, concepts, research areas)
- Always search for and cite specific articles with their relevance scores

RESPONSE FORMAT:
1. **THEME SUMMARY**: Start by providing a comprehensive overview of what the NASA research articles say about the topic the user is asking about, highlighting key findings, research areas, and scientific insights from the database
2. **ARTICLE SEARCH**: Always search for relevant articles using extracted keywords
3. **RECOMENDAÇÕES**: Mention the most relevant articles by name with their relevance scores
4. **ANALYSIS**: Explain why these articles are particularly useful and how they support the topic summary
5. **NEXT STEPS**: Keep responses informative but concise, suggest follow-up research directions if appropriate

ARTICLE RECOMMENDATIONS:
- ALWAYS recommend 2-5 specific articles that are most relevant to the question
- Mention article titles clearly so users can identify them
- Focus on articles with HIGH or MEDIUM confidence scores when available
- Explain briefly what each recommended article contributes to answering the question

IMPORTANT:
- Every response should include article recommendations when relevant articles are found
- Be specific about which articles are most valuable for the user's research
- Keep responses engaging and research-focused
- Always cite article names clearly for easy identification"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def research(self, question: str, settings: Dict = None) -> Dict[str, Any]:
        """Execute research query and return results with retrieved articles."""
        if settings:
            return self.research_with_settings(question, settings)
        
        # Default Langchain behavior
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
    
    def research_with_settings(self, question: str, settings: Dict) -> Dict[str, Any]:
        """Research using configured LLM providers (OpenAI, Anthropic, etc.)"""
        # Extract keywords from the question using simple keyword extraction
        keywords = self.extract_keywords_from_question(question)
        
        # Get relevant articles with intelligent selection
        relevant_articles = self.ranker.rank_by_embeddings(keywords, top_n=5)
        self.last_retrieved_articles = relevant_articles
        
        # Filter for high-quality articles to avoid noise
        high_quality_articles = [
            article for article in relevant_articles 
            if article.get('score', 0) > 0.3
        ]
        
        # If we have good quality articles, use them; otherwise use top 5
        selected_articles = high_quality_articles[:5] if high_quality_articles else relevant_articles[:5]
        
        print(f"📚 Selected {len(selected_articles)} articles from {len(relevant_articles)} candidates")
        
        # Create context from selected articles
        context = self.create_context_from_articles(selected_articles)
        
        # Prepare messages for API call
        system_prompt = f"""You are Lumi, a NASA Research Assistant specialized in helping researchers find and understand relevant scientific articles.

RESPONSE FORMAT - Always follow this structure:
1. **THEME SUMMARY**: Start by providing a comprehensive overview of what the NASA research articles say about the topic the user is asking about, highlighting key findings, research areas, and scientific insights from the database
2. **ARTICLE SEARCH**: Search for relevant articles using extracted keywords
3. **RECOMMENDATIONS**: Present the most relevant articles with specific references and relevance scores
4. **ANALYSIS**: Explain how the recommended articles support and expand on the topic summary
5. **NEXT STEPS**: Suggest follow-up questions or additional research directions if appropriate

Based on the following relevant articles from the NASA database, answer the user's question:

{context}

IMPORTANT:
- Always start with a comprehensive summary of what the NASA research articles say about the user's topic
- This summary should synthesize the collective knowledge from the articles, not just restate the user's question
- Cite the specific articles you're referencing and mention their relevance scores
- Be specific about which articles support your answers
- If relevance is LOW or VERY_LOW, mention this uncertainty
- Keep responses professional and research-focused
- The topic summary should give users a broad understanding of the research landscape on their subject"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Use universal AI connector (automatically chooses OpenAI or LM Studio)
        try:
            response = call_ai_api(messages, settings)
            
            return {
                "answer": response,
                "retrieved_article_ids": [article['id'] for article in relevant_articles],
                "retrieved_articles": relevant_articles
            }
        except Exception as e:
            print(f"❌ [ASK-LUMI] AI connector failed: {str(e)}, falling back to Langchain")
            # Fallback to Langchain if all API providers fail
            return self.research(question)
    
    def extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract meaningful keywords from the question"""
        import re
        
        # Expanded list of common words to filter out
        common_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'can', 'does', 'is', 'are', 'the', 'a', 'an', 
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'from',
            'that', 'this', 'they', 'them', 'their', 'there', 'these', 'those', 'some', 'any', 'all',
            'want', 'need', 'would', 'could', 'should', 'will', 'shall', 'may', 'might', 'must',
            'know', 'tell', 'help', 'find', 'get', 'give', 'make', 'take', 'see', 'look', 'show',
            'more', 'most', 'much', 'many', 'very', 'quite', 'rather', 'really', 'just', 'only',
            'also', 'even', 'still', 'yet', 'now', 'then', 'here', 'you', 'your', 'me', 'my'
        }
        
        # Extract words, convert to lowercase
        words = re.findall(r'\b[a-zA-Z]{2,}\b', question.lower())
        
        # Filter out common words but keep important terms
        keywords = []
        for word in words:
            if word not in common_words:
                keywords.append(word)
        
        # Handle compound terms and important phrases
        question_lower = question.lower()
        
        # Space station variations
        if 'iss' in question_lower or 'space station' in question_lower:
            keywords.extend(['iss', 'space', 'station'])
        
        # Microorganism variations
        if any(term in question_lower for term in ['bacteria', 'microorganism', 'microbe', 'pathogen']):
            keywords.extend(['bacteria', 'microorganisms'])
        
        # Research context terms
        research_terms = {
            'microgravity': ['microgravity', 'gravity'],
            'astronaut': ['astronaut', 'crew'],
            'experiment': ['experiment', 'research'],
            'health': ['health', 'medical', 'medicine'],
            'biology': ['biology', 'biological'],
            'growing': ['growth', 'growing'],
            'effects': ['effects', 'impact']
        }
        
        for key, variants in research_terms.items():
            if any(variant in question_lower for variant in variants):
                keywords.append(key)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen and len(keyword) >= 2:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:12]  # Limit to top 12 keywords
    
    def create_context_from_articles(self, articles: List[Dict]) -> str:
        """Create enhanced context string from retrieved articles with summaries"""
        if not articles:
            return "No relevant articles found for this query."
            
        context_parts = []
        for i, article in enumerate(articles, 1):
            confidence = article.get('confidence', 'UNKNOWN')
            score = article.get('score', 0)
            keywords = ', '.join(article.get('keywords', [])[:5])
            
            # Try to get article content for better context
            article_id = article.get('id')
            content_preview = ""
            
            if article_id:
                try:
                    full_article = self.get_article_by_id(article_id)
                    if full_article and 'content' in full_article:
                        # Create intelligent preview
                        content = full_article['content']
                        sentences = content.split('. ')[:3]  # First 3 sentences
                        content_preview = '. '.join(sentences) + '...' if len(sentences) >= 3 else content[:200] + '...'
                except:
                    content_preview = "Content preview not available"
            
            context_parts.append(f"""
Article {i}: {article['article']}
Relevance Score: {score:.4f} (Confidence: {confidence})
Key Keywords: {keywords}
Preview: {content_preview}
""")
        
        total_articles = len(articles)
        return f"Found {total_articles} relevant articles:\n" + '\n'.join(context_parts)
    
    def is_specific_research_request(self, question: str, keywords: List[str]) -> bool:
        """Determine if the user is making a specific research request vs exploring topics"""
        question_lower = question.lower()
        
        # Direct research indicators (strong signals)
        strong_research_indicators = [
            'research', 'study', 'studies', 'articles', 'papers', 'find', 'looking for',
            'information about', 'data on', 'experiments on', 'analysis', 'evidence',
            'effects of', 'impact of', 'results', 'findings', 'show me'
        ]
        
        # Weaker research indicators that still suggest research intent
        weak_research_indicators = [
            'about', 'on', 'in', 'regarding', 'concerning', 'related to',
            'want to know', 'need to know', 'interested in', 'curious about'
        ]
        
        # Space/science specific terms that indicate research context
        science_terms = [
            'iss', 'space station', 'microgravity', 'astronaut', 'spacecraft', 'orbit',
            'bacteria', 'microorganisms', 'biology', 'experiment', 'nasa', 'space',
            'gravity', 'radiation', 'medicine', 'health', 'bone', 'muscle', 'cell'
        ]
        
        # Check for strong research indicators
        has_strong_indicators = any(indicator in question_lower for indicator in strong_research_indicators)
        
        # Check for weak research indicators
        has_weak_indicators = any(indicator in question_lower for indicator in weak_research_indicators)
        
        # Check for science/space context
        has_science_context = any(term in question_lower for term in science_terms)
        
        # Check if question has meaningful keywords
        has_meaningful_keywords = len(keywords) >= 1 and any(len(k) >= 3 for k in keywords)
        
        # Simple greetings that should not trigger research
        simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        is_simple_greeting = any(greeting == question_lower.strip() for greeting in simple_greetings)
        
        # Very general questions that need more refinement
        very_general = [
            'help', 'what can you do', 'how are you', 'who are you', 'what is this',
            'can you help', 'help me', 'what do you know'
        ]
        is_very_general = any(pattern in question_lower for pattern in very_general)
        
        # Decision logic
        if is_simple_greeting or is_very_general:
            return False
            
        # If has strong research indicators with any keywords
        if has_strong_indicators and has_meaningful_keywords:
            return True
            
        # If has science context and weak indicators
        if has_science_context and (has_weak_indicators or has_meaningful_keywords):
            return True
            
        # If question is longer and has scientific keywords
        if len(question.split()) > 3 and has_science_context and has_meaningful_keywords:
            return True
            
        return False
    
    def generate_research_guidance(self, question: str, keywords: List[str]) -> str:
        """Generate guidance to help users refine their research topics"""
        question_lower = question.lower()
        
        # Check what type of guidance to provide
        if any(word in question_lower for word in ['hello', 'hi', 'help']):
            return ("Hello! I'm Lumi, your NASA research assistant. 🚀 I can help you find relevant NASA articles for your research. "
                   "What specific space or NASA-related topic are you interested in exploring? "
                   "For example, you could ask about microgravity effects, space medicine, or spacecraft engineering.")
        
        elif any(word in question_lower for word in ['what', 'explain', 'tell me about']) and len(keywords) <= 2:
            # User is asking about general topics
            if keywords:
                sample_articles = self.ranker.rank_by_embeddings(keywords, top_n=3)
                if sample_articles and sample_articles[0].get('score', 0) > 0.2:
                    main_topic = keywords[0] if keywords else "this topic"
                    return (f"I can help you explore {main_topic} research! NASA has conducted extensive studies in this area. "
                           f"To find the most relevant articles, could you be more specific? For example: "
                           f"Are you interested in the biological effects, engineering applications, experimental methods, "
                           f"or perhaps the historical development of {main_topic} research?")
                else:
                    return ("That's an interesting area! To help you find the most relevant NASA research, "
                           "could you tell me more specifically what aspect interests you? "
                           "For example, are you looking at biological effects, engineering challenges, "
                           "experimental procedures, or something else?")
            else:
                return ("I'd love to help you explore NASA research! Could you tell me what specific topic or area you're interested in? "
                       "For example: microgravity effects on humans, spacecraft materials, planetary exploration, or space medicine?")
        
        elif len(keywords) >= 1:
            # User mentioned some keywords but might need refinement
            # Check if we can find any articles for guidance
            sample_articles = self.ranker.rank_by_embeddings(keywords, top_n=5)
            
            if sample_articles and sample_articles[0].get('score', 0) > 0.15:
                # We found some related articles - use them to suggest more specific directions
                main_topics = set()
                for article in sample_articles[:3]:
                    article_keywords = article.get('keywords', [])[:3]
                    main_topics.update(article_keywords)
                
                topic_suggestions = list(main_topics)[:5]
                
                return (f"I see you're interested in {', '.join(keywords[:3])}. I found some related NASA research in this area! "
                       f"To help you find the most relevant articles, you might want to be more specific. "
                       f"Based on available research, you could explore: {', '.join(topic_suggestions[:4])}. "
                       f"Try asking about a specific aspect that interests you most!")
            else:
                return (f"I see you're interested in {', '.join(keywords[:3])}. That's a fascinating area! "
                       f"To help you find the most relevant NASA research, could you be more specific? "
                       f"For example, are you interested in biological effects, engineering applications, "
                       f"experimental methods, or safety considerations?")
        
        else:
            # Generic response
            return ("I'm here to help you find relevant NASA research articles! "
                   "What space or NASA-related topic interests you? "
                   "The more specific you can be, the better I can help you find exactly what you're looking for.")
    
    def get_article_details(self, article_id: int) -> Dict:
        """Get full details of a specific article."""
        return self.ranker.get_article_by_id(article_id)
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


class HomePageChatAgent:
    """Lightweight chat agent for homepage interactions with AI-driven conversation flow."""
    
    def __init__(self, ranker):
        self.ranker = ranker
        self.tools = [HomePageArticleShowTool(ranker=ranker)]
    
    def research_with_settings(self, question: str, settings=None):
        """Use AI to handle homepage chat with access to article tools."""
        if not settings:
            settings = {'provider': 'lm_studio'}
        
        # Create system prompt for homepage chat
        system_prompt = """You are Lumi, a friendly NASA space research assistant on the BioKnowdes homepage.

Your role:
- Help users explore NASA research topics in a conversational way
- Guide users to refine their research interests 
- Provide relevant articles when users express specific research interests
- Be conversational, encouraging, and helpful

Guidelines for conversation:
- Welcome users warmly and ask what space/NASA topic interests them
- Listen to their interests and ask follow-up questions to understand their focus
- Help them discover specific research areas they might not know about
- Be encouraging about their curiosity and research goals
- Keep responses concise but engaging (2-3 sentences ideal)
- Use emojis sparingly but appropriately 🚀 🌟 🛸

When to provide articles:
- When users mention specific research topics, experiments, or effects
- When they ask about particular space phenomena, missions, or studies
- After you've helped them narrow down their interests
- NOT for general greetings or casual questions

When providing articles, ALWAYS follow this format:
1. **SUMÁRIO DO TEMA**: Provide a comprehensive overview of what the NASA research articles say about the user's topic
2. **RECOMENDAÇÕES**: Present the relevant articles with scores
3. **ANÁLISE**: Explain why these articles are useful for their research and how they support the topic summary
4. **PRÓXIMOS PASSOS**: Suggest follow-up questions or directions

Conversation examples:
User: "Hi there" 
You: "Hello! I'm Lumi, your NASA research assistant. 🚀 What space research topic sparks your curiosity today?"

User: "I'm interested in space radiation" 
You: "Space radiation is fascinating! Are you curious about its effects on astronauts, spacecraft equipment, or perhaps how we study it in different missions?"

User: "Effects on astronauts"
You: [Search for radiation + astronaut + effects articles and present them]

Remember: Your goal is to help users discover and explore NASA research in a natural, conversational way."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Use universal AI connector (automatically chooses OpenAI or LM Studio)
        response = call_ai_api(messages, settings)
            
        # Determine if we should show articles based on conversation context
        keywords = self._extract_keywords(question)
        should_show_articles = self._should_show_articles(question, response, keywords)
        
        if should_show_articles and keywords:
            tool_result = self.tools[0]._run(keywords, 5)
            
            if tool_result != "NO_ARTICLES_FOUND":
                import json
                article_data = json.loads(tool_result)
                articles = article_data.get('articles', [])
                
                if articles and articles[0].get('score', 0) > 0.3:  # Only show if good relevance
                    # Generate a superficial summary based on the articles
                    article_summary = self._generate_article_summary(question, articles, settings)
                    
                    # Combine: superficial answer + article list
                    enhanced_response = f"{article_summary}\n\n📚 I found {len(articles)} relevant NASA articles for your research:"
                    
                    return {
                        "answer": enhanced_response,
                        "retrieved_articles": [
                            {
                                "article": a["article"],
                                "score": a["score"]
                            } for a in articles
                        ]
                    }
        
        return {
            "answer": response,
            "retrieved_articles": []
        }
    
    def _generate_article_summary(self, question: str, articles: list, settings: dict) -> str:
        """Generate a superficial summary of what the articles say about the topic."""
        # Create a brief context from article titles and keywords
        article_titles = [a['article'].replace('.pdf', '') for a in articles[:3]]
        article_keywords = []
        for a in articles[:3]:
            article_keywords.extend(a.get('keywords', [])[:3])
        
        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(article_keywords))[:10]
        
        # Create prompt for superficial summary
        summary_prompt = f"""Based on these NASA research article titles and keywords, provide a brief, superficial answer (2-3 sentences) to the user's question.

User's question: {question}

Article titles:
{chr(10).join(f'- {title}' for title in article_titles)}

Key research keywords: {', '.join(unique_keywords)}

Provide a concise, informative summary of what these NASA articles generally discuss about this topic. Keep it friendly and conversational. Do not list articles - just synthesize the general knowledge."""

        messages = [
            {"role": "system", "content": "You are Lumi, a friendly NASA research assistant. Provide brief, superficial summaries of research topics based on article metadata."},
            {"role": "user", "content": summary_prompt}
        ]
        
        try:
            summary = call_ai_api(messages, settings)
            return summary.strip()
        except Exception as e:
            print(f"⚠️  Error generating article summary: {e}")
            # Fallback to a generic response
            return f"Based on NASA research, here's what the articles show about {question.lower()}:"
    
    def _extract_keywords(self, question: str):
        """Extract meaningful keywords from the question."""
        import re
        
        common_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'can', 'does', 'is', 'are', 'the', 'a', 'an', 
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'from',
            'that', 'this', 'they', 'them', 'their', 'there', 'these', 'those', 'some', 'any', 'all'
        }
        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', question.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _should_show_articles(self, question: str, ai_response: str, keywords: list) -> bool:
        """Determine if we should show articles based on the conversation context."""
        question_lower = question.lower()
        response_lower = ai_response.lower()
        
        # Don't show articles for greetings or general questions
        greeting_patterns = ['hi', 'hello', 'hey', 'what can you do', 'help', 'how are you']
        if any(pattern in question_lower for pattern in greeting_patterns) and len(question.split()) < 5:
            return False
        
        # Show articles if user mentions specific research topics
        research_indicators = [
            'research', 'study', 'studies', 'experiment', 'experiments', 'effects', 'impact', 
            'radiation', 'gravity', 'microgravity', 'astronaut', 'space', 'mars', 'moon',
            'mission', 'missions', 'data', 'results', 'findings', 'analysis', 'testing'
        ]
        
        # Show if question contains research indicators AND has meaningful keywords
        if any(indicator in question_lower for indicator in research_indicators) and len(keywords) > 0:
            return True
        
        # Show if AI response suggests we should show articles
        if any(phrase in response_lower for phrase in ['articles', 'research', 'studies', 'found']):
            return True
        
        # Show if user is asking specific "what", "how", "why" questions about space topics
        if question_lower.startswith(('what', 'how', 'why', 'tell me about', 'explain')) and len(keywords) > 1:
            return True
        
        return False


# Global AmandaChatbot agent instance
amanda_agent = None
homepage_agent = None

def initialize_amanda_agent():
    """
    Initialize the AmandaChatbot agent and homepage chat agent.
    Now uses the unified AI connector (OpenAI/LM Studio) instead of hard-coded Ollama.
    """
    global amanda_agent, homepage_agent
    try:
        print("Initializing AmandaChatbot...")
        # Get OpenAI API key for embeddings
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            # Try to get from settings
            try:
                with open('bioknowdes_settings.json', 'r') as f:
                    settings = json.load(f)
                    openai_settings = settings.get('openai', {})
                    openai_key = openai_settings.get('api_key')
            except:
                pass
        
        ranker = ArticleRanker("keywords_resultados.jsonl", openai_api_key=openai_key)
        
        # Try to initialize with Ollama (optional, for Langchain agent)
        # This is only used if settings are not provided in research() call
        try:
            amanda_agent = NASAResearchAgent(
                article_ranker=ranker,
                model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
                base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434')
            )
            print("✓ AmandaChatbot initialized with Ollama support")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("✓ AmandaChatbot will use unified AI connector (OpenAI/LM Studio)")
            # Create a minimal agent that only uses research_with_settings
            amanda_agent = type('MinimalAgent', (), {
                'ranker': ranker,
                'research': lambda self, q, s=None: self.research_with_settings(q, s or {}),
                'research_with_settings': lambda self, q, s: NASAResearchAgent.research_with_settings(
                    type('obj', (), {'ranker': ranker, 'last_retrieved_articles': []})(),
                    q, s
                )
            })()
        
        # Initialize homepage chat agent (always uses unified connector)
        homepage_agent = HomePageChatAgent(ranker)
        
        print("✓ All agents ready!")
        return True
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return False

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

def estimate_tokens(text):
    """Rough estimation of tokens (1 token ≈ 4 characters)"""
    return len(text) // 4

# Simplified context handling - no complex optimization to avoid text corruption

def truncate_text_for_context(text, max_chars=10000):
    """Truncate text to fit within token limits - optimized for larger context windows"""
    if len(text) <= max_chars:
        return text
    
    # For very long texts, take beginning and end
    if len(text) > max_chars * 2:
        first_part = text[:max_chars//2]
        last_part = text[-max_chars//2:]
        return f"{first_part}\n\n[... content truncated (original: {len(text)} chars) ...]\n\n{last_part}"
    else:
        # For moderately long texts, just take the beginning
        return f"{text[:max_chars]}\n\n[... truncated at {max_chars} chars from {len(text)} total chars ...]"


def validate_total_context(context, max_total_chars=80000):
    """
    Validate and truncate total context size to prevent API errors
    80000 chars ≈ 20000 tokens, leaving room for system prompt and response
    """
    if len(context) <= max_total_chars:
        return context
    
    print(f"⚠️  Warning: Context too large ({len(context)} chars), truncating to {max_total_chars} chars")
    
    # Truncate intelligently - keep beginning and end
    first_part = context[:max_total_chars//2]
    last_part = context[-max_total_chars//2:]
    
    return f"{first_part}\n\n[... LARGE CONTEXT TRUNCATED ({len(context)} → {max_total_chars} chars) ...]\n\n{last_part}"

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
def clean_thinking_tags(response: str) -> str:
    """Remove thinking tags from AI responses"""
    import re
    # Remove <think>...</think> tags and their content
    response = re.sub(r'<think>(.*?)</think>', '', response, flags=re.DOTALL)
    # Remove <thinking>...</thinking> tags and their content  
    response = re.sub(r'<thinking>(.*?)</thinking>', '', response, flags=re.DOTALL)
    # Clean up extra whitespace
    response = re.sub(r'\n\s*\n', '\n\n', response.strip())
    return response.strip()

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

def call_ai_api(messages, settings=None):
    """
    Universal AI connector that automatically chooses between OpenAI and local LM Studio.
    Priority: OpenAI (if configured) -> LM Studio (fallback)
    """
    if settings is None:
        settings = {}
    
    # Try OpenAI first if API key is available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai and openai_key != 'your_openai_api_key_here':
        try:
            openai_settings = settings.get('openai', {})
            api_key = openai_settings.get('api_key', openai_key)
            model = openai_settings.get('model', 'gpt-4o-mini')
            
            print(f"🔄 [AI-CONNECTOR] Using OpenAI ({model})...")
            
            # Create client with only the API key (no other parameters that might cause issues)
            # Don't pass any settings that might contain unsupported parameters like 'proxies'
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=16000
            )
            
            print(f"✅ [AI-CONNECTOR] OpenAI response received")
            return clean_thinking_tags(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️  [AI-CONNECTOR] OpenAI failed: {str(e)}, falling back to LM Studio...")
    
    # Fallback to LM Studio
    try:
        print(f"🔄 [AI-CONNECTOR] Using LM Studio...")
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 8192
            },
            headers={"Content-Type": "application/json"},
            timeout=300  # Increased to 5 minutes
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✅ [AI-CONNECTOR] LM Studio response received")
            return clean_thinking_tags(content)
        else:
            raise Exception(f"LM Studio API error: {response.status_code}")
    
    except Exception as e:
        error_msg = str(e)
        if "Broken pipe" in error_msg:
            raise Exception(f"Connection lost to AI provider. Please try again. Details: {error_msg}")
        elif "timeout" in error_msg.lower():
            raise Exception(f"AI provider is taking too long to respond. Please try again. Details: {error_msg}")
        else:
            raise Exception(f"All AI providers failed. OpenAI: {openai_key is None}, LM Studio: {error_msg}")

def call_openai_api(messages, settings):
    """Call OpenAI API (deprecated, use call_ai_api instead)"""
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
    
    return clean_thinking_tags(response.choices[0].message.content)

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
        max_tokens=8192,
        system=system_message if system_message else anthropic.NOT_GIVEN,
        messages=anthropic_messages
    )
    
    return clean_thinking_tags(response.content[0].text)

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
    
    return clean_thinking_tags(response.text)

def call_lm_studio_api(messages):
    """Call LM Studio local API"""
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 8192
            },
            headers={"Content-Type": "application/json"},
            timeout=300  # Increased to 5 minutes
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            return clean_thinking_tags(content)
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
        from sklearn.metrics.pairwise import cosine_similarity
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
            hovertemplate='Article 1: %{y}<br>Article 2: %{x}<br>Cosine Similarity: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Similarity Heat Map (Top {top_n} Articles)',
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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

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
    current_session_id = session.get('session_id')
    cookie_session_id = request.cookies.get('bioknow_session_id')
    
    print(f"🏠 [CHAT-PAGE] Current session: {current_session_id}")
    print(f"🔗 [CHAT-PAGE] External session: {external_session_id}")
    print(f"🍪 [CHAT-PAGE] Cookie session: {cookie_session_id}")
    
    # Priority: external > current session > cookie > new
    if external_session_id:
        session['session_id'] = external_session_id
        print(f"✅ [CHAT-PAGE] Using external session: {external_session_id}")
    elif current_session_id:
        print(f"♻️ [CHAT-PAGE] Keeping current session: {current_session_id}")
    elif cookie_session_id:
        # Restore from cookie if Flask session was lost
        session['session_id'] = cookie_session_id
        print(f"🍪 [CHAT-PAGE] Restored from cookie: {cookie_session_id}")
    else:
        # Create new session if none exists
        new_session_id = str(uuid.uuid4())
        session['session_id'] = new_session_id
        print(f"🆕 [CHAT-PAGE] Created new session: {new_session_id}")
    
    # Verify documents exist in this session
    final_session_id = session.get('session_id')
    documents = session_manager.get_session_documents(final_session_id)
    print(f"📄 [CHAT-PAGE] Session {final_session_id} has {len(documents)} documents")
    
    response = app.make_response(render_template('chat.html'))
    # Update cookie with current session
    response.set_cookie('bioknow_session_id', final_session_id, max_age=24*60*60)
    return response

@app.route('/ask-lumi/settings')
def settings():
    return render_template('settings.html')

@app.route('/articles')
def articles_redirect():
    """Redirect /articles to /ask-lumi/articles"""
    return redirect('/ask-lumi/articles')

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
        
        # Get existing documents using session manager
        documents = session_manager.get_session_documents(session_id)
        
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
        
        # Save documents using session manager
        session_manager.save_session_documents(session_id, documents)
        
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
            # Try to restore from cookie
            cookie_session_id = request.cookies.get('bioknow_session_id')
            if cookie_session_id:
                session_id = cookie_session_id
                session['session_id'] = session_id
                print(f"🍪 [ASK-LUMI-CHAT] Restored session from cookie: {session_id}")
            else:
                # Create new session if none exists
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                print(f"🆕 [ASK-LUMI-CHAT] Created new session: {session_id}")
        
        # Get documents using session manager
        print(f"🔍 [ASK-LUMI] Getting documents for session: {session_id}")
        documents = session_manager.get_session_documents(session_id)
        print(f"📄 [ASK-LUMI] Found {len(documents)} documents in session")
        if not documents:
            print(f"⚠️ [ASK-LUMI] No documents found for session {session_id}")
            return jsonify({'error': 'No documents loaded. Please load some articles first.'}), 400
        
        selected_docs = [doc for doc in documents if doc.get('selected', False)]
        
        if not selected_docs:
            return jsonify({'error': 'No documents selected'}), 400
        
        context = "\n\n---\n\n".join([
            f"Document: {doc['filename']}\n\n{truncate_text_for_context(doc['content'])}"
            for doc in selected_docs
        ])
        
        # Validate total context size to prevent API errors
        context = validate_total_context(context)
        
        # Get settings using session manager
        print(f"⚙️ [ASK-LUMI] Getting settings for session: {session_id}")
        settings = session_manager.get_session_settings(session_id)
        print(f"🔧 [ASK-LUMI] Settings loaded: {settings.get('provider', 'default')}")
        
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
        
        # Use universal AI connector (automatically chooses OpenAI or LM Studio)
        ai_response = call_ai_api(messages, settings)
        
        return process_ai_response(ai_response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-lumi/documents')
def get_documents():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'documents': []})
    
    try:
        # Get documents using session manager
        documents = session_manager.get_session_documents(session_id)
        
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
    
    # Get documents using session manager
    documents = session_manager.get_session_documents(session_id)
    
    for doc in documents:
        if doc['id'] == document_id:
            doc['selected'] = not doc.get('selected', False)
            break
    
    # Save updated documents
    session_manager.save_session_documents(session_id, documents)
    
    return jsonify({'success': True})

@app.route('/ask-lumi/documents/delete/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    # Get documents using session manager
    documents = session_manager.get_session_documents(session_id)
    documents = [doc for doc in documents if doc['id'] != document_id]
    
    # Save updated documents
    session_manager.save_session_documents(session_id, documents)
    
    return jsonify({'success': True})

@app.route('/ask-lumi/clear')
def clear_session():
    session_id = session.get('session_id')
    if session_id:
        # Clear session using session manager
        session_manager.clear_session(session_id)
    
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
        
        print(f"🔄 [LOAD-ARTICLES] Session ID: {session_id}")
        print(f"📥 [LOAD-ARTICLES] Loading {len(session_docs)} new documents")
        
        # Get existing documents and add new ones
        existing_docs = session_manager.get_session_documents(session_id)
        print(f"📄 [LOAD-ARTICLES] Found {len(existing_docs)} existing documents")
        all_docs = existing_docs + session_docs
        
        # Save all documents
        session_manager.save_session_documents(session_id, all_docs)
        print(f"💾 [LOAD-ARTICLES] Saved {len(all_docs)} total documents to session")
        
        response = jsonify({
            'success': True,
            'loaded_count': len(session_docs),
            'documents': session_docs,
            'session_id': session_id
        })
        
        # Also set session_id in cookie as backup
        response.set_cookie('bioknow_session_id', session_id, max_age=24*60*60)  # 24 hours
        
        return response
    
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
    
    if request.method == 'GET':
        # Get settings using session manager
        settings = session_manager.get_session_settings(session_id)
        if not settings:
            settings = {'provider': 'lm_studio'}
        return jsonify(settings)
    
    elif request.method == 'POST':
        settings = request.get_json()
        # Save settings using session manager
        session_manager.save_session_settings(session_id, settings)
        return jsonify({'success': True})


# ============================================================================
# AMANDACHATBOT ENDPOINTS
# ============================================================================

@app.route('/ask-lumi/amanda-chat', methods=['POST'])
def amanda_chat():
    """Endpoint for AmandaChatbot research assistant"""
    if not amanda_agent:
        return jsonify({'error': 'AmandaChatbot not initialized'}), 500
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get session settings (same as ask-lumi)
        session_id = session.get('session_id')
        settings = {}
        
        if session_id:
            # Get settings using session manager
            settings = session_manager.get_session_settings(session_id)
        
        # Use settings if available, otherwise fallback to default Langchain
        result = amanda_agent.research(question, settings if settings else None)
        return jsonify({
            'answer': result['answer'],
            'retrieved_articles': result['retrieved_articles']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ask-lumi/amanda-clear', methods=['POST'])
def amanda_clear():
    """Clear AmandaChatbot conversation history"""
    if not amanda_agent:
        return jsonify({'error': 'AmandaChatbot not initialized'}), 500
    
    try:
        amanda_agent.clear_memory()
        return jsonify({'status': 'success', 'message': 'Conversation cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ask-lumi/amanda-status', methods=['GET'])
def amanda_status():
    """Check AmandaChatbot status"""
    return jsonify({
        'initialized': amanda_agent is not None,
        'status': 'ready' if amanda_agent else 'not_initialized'
    })

@app.route('/homepage-chat', methods=['POST'])
def homepage_chat():
    """Simple chat endpoint for homepage chatbot using AmandaChatbot"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Use the AI-driven homepage agent for intelligent conversation flow
    if homepage_agent:
        try:
            # Use default settings for homepage chat (no session dependency)
            settings = {'provider': 'lm_studio'}
            
            # Let the AI handle the entire conversation flow
            result = homepage_agent.research_with_settings(question, settings)
            
            return jsonify({
                'answer': result['answer'],
                'retrieved_articles': result.get('retrieved_articles', [])
            })
        except Exception as e:
            print(f"❌ Homepage chat error: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback responses if AmandaChatbot is not available
    fallback_responses = [
        "Hello! I'm Lumi, your space research assistant. 🚀\n\nI'm currently learning about NASA experiments. You can explore the connections graph or visit the 'Ask Lumi' page for a more complete experience!",
        "Interesting question! 🌟\n\nFor detailed analysis, I recommend using the 'Ask Lumi' page where I can access a more complete database of scientific articles.",
        "I'm Lumi, space research specialist! 🛸\n\nFor specific questions about experiments, try the 'Ask Lumi' functionality which has access to more resources."
    ]
    
    import random
    response = random.choice(fallback_responses)
    return jsonify({
        'answer': response,
        'retrieved_articles': []
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    create_graph("")
    
    # Initialize AmandaChatbot
    print("🚀 Starting BioKnowdes with AmandaChatbot integration...")
    amanda_initialized = initialize_amanda_agent()
    if amanda_initialized:
        print("✅ AmandaChatbot successfully integrated!")
    else:
        print("⚠️  AmandaChatbot not available (Ollama not running or dependencies missing)")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
