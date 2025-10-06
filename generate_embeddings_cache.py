#!/usr/bin/env python3
"""
Generate embeddings cache for all articles using OpenAI API.
This cache can then be uploaded to a public URL for Heroku to download.
"""

import os
import json
from app import ArticleRanker

def main():
    # Get OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    print("🚀 Starting embeddings cache generation...")
    print("⚠️  This will make API calls to OpenAI - ensure you have credits!")
    
    # Initialize ranker (this will generate the cache)
    ranker = ArticleRanker("keywords_resultados.jsonl", openai_api_key=openai_key)
    
    # Cache file should now exist
    cache_file = "article_embeddings_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        print(f"✅ Cache generated successfully!")
        print(f"📊 Total embeddings: {len(cache_data)}")
        print(f"📁 File size: {os.path.getsize(cache_file) / (1024*1024):.1f} MB")
        print(f"📄 Cache file: {cache_file}")
        print()
        print("📤 Next steps:")
        print("1. Upload this file to a public URL (GitHub raw, Google Drive, etc.)")
        print("2. Set EMBEDDINGS_CACHE_URL environment variable in Heroku")
        print("3. Example: EMBEDDINGS_CACHE_URL=https://raw.githubusercontent.com/user/repo/main/cache.json")
        
    else:
        print("❌ Cache file was not created")

if __name__ == "__main__":
    main()