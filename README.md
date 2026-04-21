---
BMO 🤖

"I am BMO. I am a little computer friend."

BMO is a local LLM CLI agent built on Ollama, inspired by the loveable robot from Adventure Time. It runs entirely on your machine —
no cloud, no data leaving your device, just you and your little computer friend.

---
Features

- Conversational AI via local Ollama models
- Searches your Obsidian vault using RAG (ChromaDB + nomic-embed-text)
- Runs shell commands and reads files with your approval
- Reasoning mode toggle for harder problems
- Spinner animations and BMO personality baked in
- Input sanitisation, rate limiting, and conversation history management

---
Prerequisites

- Python 3.10+
- Ollama installed and running
- The following models pulled:
ollama pull qwen2.5-coder:14b
ollama pull deepseek-r1:14b
ollama pull nomic-embed-text

---
Setup

# Clone the repo
git clone git@github.com:YOUR_USERNAME/bmo.git
cd bmo

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.yaml.example config.yaml

Edit config.yaml to set your Obsidian vault path and Ollama host.

---
Running BMO

bmo

Or directly:

source venv/bin/activate
python3 main.py

---
Slash Commands

Command:

| /help   │ Show available commands                    
│ /clear  │ Clear conversation history                
│ /model  │ Show current model and reasoning mode status
│ /reason │ Toggle reasoning model (deepseek-r1:14b)    
│ /exit   │ Say goodbye to BMO                           

---
Configuration


Key                   |       Description                  
│ model.primary       │ Main model for most tasks                    
│ model.reasoning     │ Model used when /reason is active            
│ ollama.host         │ Ollama API address (default: localhost:11434)
│ rag.vault_path      │ Path to your Obsidian vault                  
│ rag.enabled         │ Enable/disable vault search                   
│ options.temperature │ Response creativity (0 = focused, 1.6 = wild) 

---
Roadmap

- Phase 1 — CLI loop, streaming, slash commands, BMO UI
- Phase 2 — Input sanitisation, history limits, rate limiting
- Phase 3 — Tool use (run_command, read_file) with approval flow
- Phase 4 — RAG via Obsidian vault + ChromaDB
- Phase 5 — Write capabilities, session logging, polish

---
Notes

- config.yaml is gitignored — never commit it
- chroma_db/ is gitignored — rebuild locally with python ingest.py
- BMO will never run a command or read a file without asking first

---
Mathematical!