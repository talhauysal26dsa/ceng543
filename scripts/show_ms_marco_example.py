"""MS MARCO dataset örneğini göster - Query, Document ve Qrels ilişkisi"""
import json

# Dosyaları yükle
queries = json.load(open('data/processed/ms_marco_queries_train.json', 'r', encoding='utf-8'))
qrels = json.load(open('data/processed/ms_marco_qrels_train.json', 'r', encoding='utf-8'))
docs = json.load(open('data/processed/ms_marco_documents.json', 'r', encoding='utf-8'))

# Bir query seç (ilk query'den başla)
query_id = '1185869'
query = [q for q in queries if q['query_id'] == query_id]
if not query:
    query_id = queries[0]['query_id']
    query = queries[0]
else:
    query = query[0]

# Bu query için ilgili dokümanları bul
relevant_qrels = [q for q in qrels if q['query_id'] == query_id]
relevant_doc_ids = [q['doc_id'] for q in relevant_qrels]

# İlgili dokümanları getir
relevant_docs = []
for doc_id in relevant_doc_ids[:3]:  # İlk 3 tanesini göster
    doc = [d for d in docs if d['doc_id'] == doc_id]
    if doc:
        relevant_docs.append(doc[0])

print("=" * 80)
print("MS MARCO DATASET - ORNEK ILISKI")
print("=" * 80)
print(f"\n1. QUERY (Soru):")
print(f"   Query ID: {query['query_id']}")
print(f"   Query Text: '{query['query_text']}'")
print(f"   Split: {query['split']}")

print(f"\n2. QRELS (Ground Truth - Dogru Cevap):")
print(f"   Bu query için {len(relevant_doc_ids)} ilgili dokuman var:")
for qrel in relevant_qrels[:5]:
    print(f"   - Doc ID: {qrel['doc_id']} (Relevance: {qrel['relevance']})")

print(f"\n3. DOCUMENTS (Ilgili Dokumanlarin Icerigi):")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n   Dokuman {i} (Doc ID: {doc['doc_id']}):")
    print(f"   {doc['text'][:200]}...")
    print(f"   Length: {doc['length']} words")

print("\n" + "=" * 80)
print("BUNU NASIL KULLANIRIZ?")
print("=" * 80)
print("\n1. Retriever'a query ver -> Retriever dokumanlari bulur")
print("2. Qrels ile karsilastir -> Retriever dogru dokumanlari buldu mu?")
print("3. Recall@K hesapla -> Top-K'tan kac tanesi gercekten ilgili?")

