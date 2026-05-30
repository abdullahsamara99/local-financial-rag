# main.py
import sys
import os
from src.embeddings import BilingualEmbedder
from src.vector_store import LocalVectorStore
from src.rag_pipeline import LocalRAGPipeline
from src.ingestion import DataIngestor

def init_system():
    print("🚀 Initializing the fully local bilingual system...")
    embedder = BilingualEmbedder()
    vector_store = LocalVectorStore()
    ingestor = DataIngestor()
    rag_pipeline = LocalRAGPipeline()
    return embedder, vector_store, ingestor, rag_pipeline

def main():
    embedder, vector_store, ingestor, rag_pipeline = init_system()
    
    print("\n💡 If you want to ingest a new PDF file, place it inside the project folder and type its name here.")
    print("💡 To start asking questions directly (if you have already ingested files before), just press Enter.")
    
    pdf_to_inject = input("\n📥 Do you want to ingest a PDF file? (Type its name with extension or press Enter to skip): ").strip()
    if pdf_to_inject and os.path.exists(pdf_to_inject):
        ingestor.process_pdf(pdf_to_inject)
    elif pdf_to_inject:
        print("❌ File not found. Proceeding directly to the Q&A interface.")

    print("\n🤖 The system is ready to answer your questions about the ingested documents (type 'exit' to quit):")
    
    while True:
        try:
            query = input("\n👤 Your question: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("👋 See you later! System closed.")
                break
            if not query:
                continue

            query_embedding = embedder.embed_query(query)
            matched_docs = vector_store.similarity_search(query_embedding, k=4)
            
            if not matched_docs or matched_docs[0]['score'] < 0.10:
                print("🤖 Bot: The requested information is not available in the local documents.")
                continue

            answer = rag_pipeline.generate_answer(query, matched_docs)
            
            print(f"\n🤖 Answer:\n{answer.strip()}")
            print("\n📌 Sources and Documentation:")
            for doc in matched_docs:
                print(f"   - [Page {doc['metadata'].get('page')}] | (Semantic Match Score: {doc['score']:.2%})")
                
        except KeyboardInterrupt:
            print("\n👋 Session terminated.")
            sys.exit(0)

if __name__ == "__main__":
    main()