"""
FULL PIPELINE: BM25 + Ranker + Gemini
Manuel calistirin - ben API kullanmiyorum!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
from pathlib import Path
import requests

# ============== KONFIGURASYON ==============
GEMINI_API_KEY = "AIzaSyDMNaWRXb7B8S_VOAE7Pdomg4s8q9IaGfs"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

print("=" * 70)
print("FULL PIPELINE: BM25 + RANKER + GEMINI")
print("=" * 70)

# ============== 1. LOAD COMPONENTS ==============
print("\n[1/5] Loading components...")
from src.retrievers.bm25_retriever import BM25Retriever
from src.rankers.cross_encoder_ranker import CrossEncoderRanker

print("      [OK] Components loaded")

# ============== 2. LOAD DATA ==============
print("\n[2/5] Loading documents...")
docs_file = Path('data/processed/ms_marco_documents.json')
with open(docs_file, 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Use 5000 docs for better results
test_size = 5000
documents_sample = documents[:test_size]
print(f"      [OK] {test_size:,} documents loaded")

# ============== 3. RETRIEVAL ==============
print("\n[3/5] Retrieval (BM25)...")
query = "What was the Manhattan Project?"
print(f"      Query: '{query}'")

bm25 = BM25Retriever({"k1": 1.2, "b": 0.75})
bm25.index_documents(documents_sample)

start = time.time()
retrieval_result = bm25.retrieve(query, max_documents=100)
retrieval_time = time.time() - start

print(f"      [OK] {len(retrieval_result.documents)} docs retrieved ({retrieval_time*1000:.0f}ms)")

# ============== 4. RANKING ==============
print("\n[4/5] Ranking (Cross-Encoder)...")
ranker = CrossEncoderRanker({"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"})

start = time.time()
ranking_result = ranker.rank(query, retrieval_result.documents[:20])
ranking_time = time.time() - start

print(f"      [OK] Top-20 re-ranked ({ranking_time*1000:.0f}ms)")

# Show top 3
print("\n      Top-3 Documents:")
for i, doc in enumerate(ranking_result.documents[:3], 1):
    snippet = doc['text'][:60].replace('\n', ' ')
    print(f"        [{i}] {snippet}...")

# ============== 5. GENERATION (GEMINI) ==============
print("\n[5/5] Generation (Gemini API)...")

# Build prompt with FEW-SHOT example for citations
top_docs = ranking_result.documents[:5]  # Use top 5

# Create evidence text
evidence_text = ""
for i, doc in enumerate(top_docs, 1):
    doc_text = doc['text'][:400]  # Limit length
    evidence_text += f"\n[{i}] {doc_text}\n"

prompt = f"""You are an AI that answers questions with citations.

EXAMPLE:
Question: When was the Eiffel Tower built?
Evidence:
[1] The Eiffel Tower is in Paris, France.
[2] It was constructed in 1889 for the World's Fair.
[3] The tower stands 330 meters tall.

Answer: The Eiffel Tower was built in 1889 [2] in Paris, France [1].

===== NOW YOUR TURN =====

Question: {query}

Evidence:{evidence_text}

CRITICAL: Your answer MUST follow this format:
"[Your answer with [1], [2] citations included]"

Answer:"""

try:
    start = time.time()
    
    response = requests.post(
        GEMINI_URL,
        headers={
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        },
        json={
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 512
            }
        },
        timeout=30
    )
    
    generation_time = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        answer = result['candidates'][0]['content']['parts'][0]['text']
        
        print(f"      [OK] Answer generated ({generation_time*1000:.0f}ms)")
        
        # ============== RESULTS ==============
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nQuery: {query}")
        print(f"\nAnswer:\n{answer}")
        print("\n" + "=" * 70)
        
        # Performance
        total_time = retrieval_time + ranking_time + generation_time
        print("\nPERFORMANCE:")
        print(f"  Retrieval: {retrieval_time*1000:.0f}ms")
        print(f"  Ranking:   {ranking_time*1000:.0f}ms")
        print(f"  Generation: {generation_time*1000:.0f}ms")
        print(f"  TOTAL:     {total_time*1000:.0f}ms")
        
        # Check citations
        citations = sum(1 for i in range(1, 10) if f'[{i}]' in answer)
        print(f"\nCitations found: {citations}")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Pipeline completed!")
        print("=" * 70)
        
    else:
        print(f"\n[ERROR] Gemini API failed: {response.status_code}")
        print(f"Message: {response.text[:200]}")
        if response.status_code == 429:
            print("\n[INFO] Quota exceeded - wait a few minutes and try again")
        
except Exception as e:
    print(f"\n[ERROR] Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[DONE] Test complete. Check results above.")
