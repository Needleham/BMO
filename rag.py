import os 
import re
import ollama
import chromadb
import yaml

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
        self.memory_path = os.path.expanduser(rag_cfg.get('memory_path', ""))
        # Separate collection from BMO's own saved notes
        self.memory_collection = self.client.get_or_create_collection(
            name="bmo_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts):
        """Get embeddings from Ollama for a list of text strings"""
        client = ollama.Client(host=self.ollama_host)
        embeddings = []
        for text in texts:
            response = client.embeddings(model=self.embedding_model, prompt= text[:5000])
            embeddings.append(response["embedding"])
        return embeddings
    
    def _chunk_text(self, text, source):
      """Split markdown on any heading (#-####), carrying full breadcrumb context.
      Falls back to paragraph splitting for oversized sections.
      Returns list of (enriched_chunk, doc_id) tuples."""
      folder = os.path.dirname(source)
      note_title = os.path.splitext(os.path.basename(source))[0]
      heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$')

      lines = text.split('\n')
      heading_stack = []   # [(level, heading_text), ...]
      current_lines = []
      chunks = []
      chunk_index = [0]    # list so the nested function can mutate it

      def emit(content_lines):
          content = '\n'.join(content_lines).strip()
          if not content:
              return

          # Build prefix — Topic, Note, and optional Section breadcrumb
          breadcrumb = ' > '.join(h[1] for h in heading_stack)
          prefix = f"Topic: {folder}\nNote: {note_title}\n"
          if breadcrumb:
              prefix += f"Section: {breadcrumb}\n"
          prefix += "\n"

          # Section fits within chunk_size — emit as single chunk
          if len(content) <= self.chunk_size:
              doc_id = f"{source}::{chunk_index[0]}"
              chunk_index[0] += 1
              chunks.append((prefix + content, doc_id))
              return

          # Oversized section: accumulate paragraphs up to chunk_size
          paragraphs = [p for p in content.split('\n\n') if p.strip()]
          buffer = []
          for para in paragraphs:
              buffer.append(para)
              if len('\n\n'.join(buffer)) > self.chunk_size:
                  if len(buffer) > 1:
                      # Flush all but the paragraph that tipped the limit
                      doc_id = f"{source}::{chunk_index[0]}"
                      chunk_index[0] += 1
                      chunks.append((prefix + '\n\n'.join(buffer[:-1]), doc_id))
                      buffer = [para]
                  else:
                      # Single paragraph exceeds chunk size - hard coded limit
                      for j in range(0, len(para), self.chunk_size):
                          piece = para[j:j + self.chunk_size].strip()
                          if piece:
                              doc_id = f"{source}::{chunk_index[0]}"
                              chunk_index[0] += 1
                              chunks.append((prefix + para, doc_id))
                      buffer = []
          # Flush any remaining paragraphs after the loop
          if buffer:
              doc_id = f"{source}::{chunk_index[0]}"
              chunk_index[0] += 1
              chunks.append((prefix + '\n\n'.join(buffer), doc_id))

      for line in lines:
          match = heading_pattern.match(line)
          if match:
              emit(current_lines)          # flush the previous section
              current_lines = []

              level = len(match.group(1))
              heading_text = match.group(2).strip()

              # Pop headings at the same level or deeper before pushing new one
              while heading_stack and heading_stack[-1][0] >= level:
                  heading_stack.pop()
              heading_stack.append((level, heading_text))
              current_lines = [line]       # include the heading line in the chunk
          else:
              current_lines.append(line)

      emit(current_lines)  # flush the final section
      return chunks
    
    def ingest(self, clear=False):
        """Walk vault, embed all md files, upsert into Chromadb"""
        if not os.path.exists(self.vault_path):
            print(f"\n [RAG] Vault not found at {self.vault_path} - skipping ingest.")
            return
        all_chunks, all_ids, all_metas = [], [], []

        #Clear stale data on each ingest
        if clear:
            self.client.delete_collection("bmo_vault")
            self.collection = self.client.get_or_create_collection(
                name="bmo_vault",
                metadata={"hnsw:space": "cosine"}
            )

        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden folders like .obsidian and .git
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "BMO Learning"]
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
                frontmatter = self._read_frontmatter(filepath)
                file_chunks = list(self._chunk_text(text, rel_path))
                for chunk_text, doc_id in file_chunks:
                    all_chunks.append(chunk_text)
                    all_ids.append(doc_id)
                    meta = ({"source": rel_path})
                    tags = frontmatter.get("tags", [])
                    if isinstance(tags, list):
                        meta["tags"] = ", ".join(str(t) for t in tags)
                    elif isinstance(tags, str):
                        meta["tags"] = tags
                    title = frontmatter.get("title", "")
                    if title:
                        meta["title"] = str(title)
                    all_metas.append(meta)

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

    def ingest_memory(self):
        """Index BMO Learning notes into the bmo_memory collection"""
        if not self.memory_path or not os.path.exists(self.memory_path):
            return
        all_chunks, all_ids, all_metas = [], [], []
        for root, dirs, files in os.walk(self.memory_path):
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
                rel_path = os.path.relpath(filepath, self.memory_path)
                file_chunks = list(self._chunk_text(text,rel_path))
                for chunk_text, doc_id in file_chunks:
                    all_chunks.append(chunk_text)
                    all_ids.append(f"memory::{doc_id}")
                    all_metas.append({"source": rel_path})
        if not all_chunks:
            return
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

    def _read_frontmatter(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line != '---':
                    return{}
                lines = []
                for line in f:
                    if line.strip() == '---':
                        break
                    lines.append(line)
            return yaml.safe_load(''.join(lines)) or {}
        except Exception:
            return {}

    def search(self, query, top_k=None):
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
    
    def search_memory(self, query, top_k=None):
        """ Search BMO's own saved notes in bmo_memory"""
        if top_k is None:
            top_k = self.top_k
        if self.memory_collection.count() == 0:
            return []
        query_embedding = self._embed([query])[0]
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.memory_collection.count())
        )
        return [
            (doc, meta["source"])
            for doc, meta in zip(results["documents"][0], resluts["metadatas"][0])
        ]
    
    def save_note(self, filename, content, mode="write"):
        """Write a note to BMO Learning folder and re-index memory"""
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = os.path.join(self.memory_path, filename)
        os.makedirs(self.memory_path, exist_ok=True)
        try:
            if mode == "append":
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write("\n\n" + content)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
            self.ingest_memory() # reindec immediately after save
            return f"Note saved: {filename}"
        except Exception as e:
            return f"Error saving note: {str(e)}"
    