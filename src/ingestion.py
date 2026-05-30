# src/ingestion.py
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import BilingualEmbedder
from src.vector_store import LocalVectorStore

class DataIngestor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embedder = BilingualEmbedder()
        self.vector_store = LocalVectorStore()

    def process_pdf(self, pdf_path: str):
        print(f"📂 Processing and reading PDF file: {pdf_path}")
        reader = PdfReader(pdf_path)
        
        raw_chunks = []
        metadatas = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text.strip():
                continue
                
            page_chunks = self.text_splitter.split_text(text)
            for chunk in page_chunks:
                raw_chunks.append(chunk)
                metadatas.append({"source": pdf_path, "page": page_num})

        if not raw_chunks:
            print("⚠️ No readable text found inside the file.")
            return

        print(f"✂️ The file has been split into {len(raw_chunks)} text chunks.")
        embeddings = self.embedder.embed_documents(raw_chunks)
        self.vector_store.add_documents(raw_chunks, metadatas, embeddings)
        print("✅ Document vectors successfully saved to the local database!")