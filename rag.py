import os 
import ollama
import chromadb

class RAGStore:
    def __init__(self, config):
        rag_cfg = config.get("rag", {})
        # Vault and storage paths from config
        self.vault_path = os.path.expanduser(rag_cfg.get("vault_path", ""))
        self.chroma_path = rag_cfg.get("chroma_path", "./chroma_db")
        self.embedding_model = rag_cfg.get("embedding_model", "nomic-embed-text")
        self.chunk_size = rag_cfg.get("chunk_size", 500)
        self.top_k = rag_cfg.get("top_k", 3)
        self.ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")

        # Connect to persistent local ChromaDB (creates folder if it doesn't exist)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="bmo_vault",
            metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
        )

    def _embed(self, texts):
        """Get embeddings from Ollama for a list of text strings"""
        client = ollama.Client(host=self.ollama_host)
        embeddings = []
        for text in texts:
            response = client.embeddings(model=self.embedding_model, prompt= text)
            embeddings.append(response["embedding"])
        return embeddings
    
    def _chunk_text(self, text, source):
        """Split text into fixe-size chunks, retunrs list of (chunk, doc_id) tuples"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                doc_id = f"{source}::{i}" #Unique id = file path + offset
                chunks.append((chunk, doc_id))
        return chunks
    
    def ingest(self):
        """Walk vault, embed all md files, upsert into Chromadb"""
        if not os.path.exists(self.vault_path):
            print(f"\n [RAG] Vault not found at {self.vault_path} - skipping ingest.")
            return
        all_chunks, all_ids, all_metas = [], [], []

        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden folders like .obsidian and .git
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    continue

                rel_path = os.path.relpath(filepath, self.vault_path)
                for chunk_text, doc_id in self._chunk_text(text, rel_path):
                    all_chunks.append(chunk_text)
                    all_ids.append(doc_id)
                    all_metas.append({"source": rel_path})

        if not all_chunks:
            return
        
        # Embed in batches to avoid memory spikes
        BATCH_SIZE = 50
        all_embeddings = []
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            all_embeddings.extend(self._embed(batch))
        
        # Upsert safely handles both new files and updated files
        self.collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metas
        )

    def search(self, query, top_k=None):
        """Search vault for chunks relevant to query, returns list of (text, source) tuples"""
        if top_k is None:
            top_k = self.top_k

        # Nothing indexed yet
        if self.collection.count() == 0:
            return []
        
        query_embedding = self._embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        # Zip documents with their source file metadata
        return [
            (doc, meta["source"])
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    