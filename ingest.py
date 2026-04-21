import yaml
from rag import RAGStore

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

if __name__ == "__main__":
    config = load_config()
    store = RAGStore(config)
    print(" [RAG] Starting manual ingest...")
    store.ingest()
    print(" [RAG] Ingest complete.")

    

