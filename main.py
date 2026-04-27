# main.py - BMO entry point and CLI loop

import yaml
import sys
from ui import show_startup, print_user, print_bmo_start, print_bmo_end, print_error, print_help, Spinner
from agent import BMOAgent
import time
import threading
from rag import RAGStore


def load_config(path="config.yaml"):
    """Load config.yaml from directory"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[!] config.yaml not found. Copy config.yaml.example to config.yaml first ")
        sys.exit(1)


def main():
    config = load_config()
    rag_store = None
    if config.get("rag", {}).get("enabled", False):
        rag_store = RAGStore(config)
        #Re-ingest vault in background - vault is git synced so changes frequently
        def _background_ingest():
            rag_store.ingest()
            rag_store.ingest_memory()
        thread = threading.Thread(target=_background_ingest, daemon=True)
        thread.start()
    agent = BMOAgent(config, rag_store=rag_store)
                  

    # Show BMO ASCII and start up info if enabled
    if config["ui"]["show_bmo"]:
        show_startup(config["model"]["primary"])
    use_reasoning = False
    # Input sanitisation 
    MAX_INPUT_LENGTH = 2000

    last_request_time = 0
    RATE_LIMIT_SECONDS = 2

    while True:
        try:
            user_input = input("\n > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n Bye!\n")
            break
        if not user_input:
            continue
        if user_input == "/exit":
            print("\n Bye!\n")
            break
        elif user_input == "/help":
            print_help()
            continue
        elif user_input == "/clear":
            agent.clear_history()
            print("\n History Cleared. \n")
            continue
        elif user_input == "/model":
            model = agent.reasoning_model if use_reasoning else agent.primary_model
            print(f"\n Reasoning mode: {use_reasoning}\n")
            continue
        elif user_input == "/reason":
            use_reasoning = not use_reasoning
            print(f"\n Reasoning mode: {'on' if use_reasoning else 'off'}\n")
            continue
        if len(user_input) > MAX_INPUT_LENGTH:
            print(f"\n [BMO] Input too long ({len(user_input)} chars). Max is {MAX_INPUT_LENGTH}.")
            continue
        now = time.time()
        if now - last_request_time < RATE_LIMIT_SECONDS:
            print(f"\n [BMO] Slow down - wait {RATE_LIMIT_SECONDS} seconds between messages.")
            continue
              
        #Send to agent and stream response
        print_user(user_input)
        print_bmo_start()
        spinner = Spinner()
        spinner.start()
        first_token = True
        try:
            for token in agent.chat(user_input, use_reasoning=use_reasoning):
                if first_token:
                    spinner.stop()
                    first_token = False
                sys.stdout.write(token)
                sys.stdout.flush()
            if first_token:
                spinner.stop()
            last_request_time = time.time()
            print_bmo_end()
        except Exception as e:
            spinner.stop()
            print_error(f"Ollama error: {e}")

if __name__ == "__main__":
    main()