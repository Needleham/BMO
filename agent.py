# agent.py - Ollam APU wrapper for BMO

import ollama

class BMOAgent:
  TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "run_command",
      "description": "Run a shell command and return the output",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The shell command to run"
          }
        },
        "required": ["command"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the contents of a file at a given path",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Absolute or relative path to the file"
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search_notes",
      "description": "Search the local Obsidian vault for notes relevant to a query",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query to find relevant notes"
          }
        },
        "required": ["query"]
      }
    }
  }
]
  def __init__(self, config, rag_store=None):
    # pull model nmes and ollama host from config
    self.primary_model = config["model"]["primary"]
    self.reasoning_model = config["model"]["reasoning"]
    self.host = config["ollama"]["host"]
    self.stream = config["ui"]["stream"]
    self.system_prompt = config["system_prompt"]
    self.config = config
    self.rag_store = rag_store 

    # initalise conversation history
    self.history = []

    # set up ollama client pointing  at our host 
    self.client = ollama.Client(host=self.host)

  def chat(self , user_input, use_reasoning=False):
    """ Send a message and yield a response chunks if streaming """
    model = self.reasoning_model if use_reasoning else self.primary_model
    # pull temp / top_p from config file
    options = self.config.get("options", {})


    # append user message to history
    self.history.append({"role": "user", "content": user_input})
    # History Size Limit
    MAX_HISTORY_MESSAGES = 20
    if len(self.history) > MAX_HISTORY_MESSAGES:
      self.history = self.history[-MAX_HISTORY_MESSAGES:] #keeps most recent

    # build full message list with system prompt first
    messages = [
      {"role": "system", "content": self.system_prompt}
    ] + self.history

    if self.stream:
      response = self.client.chat(model=model, messages=messages, tools=self.TOOLS, stream=False, options=options)
      final_message = response.message
      if final_message.tool_calls:
        yield ""
        yield from self._handle_tool_calls(final_message.tool_calls, messages, model, options)
      else:
        full_response = final_message.content or ""
        yield full_response
        self.history.append({"role": "assistant", "content": full_response})
    else:
      response = self.client.chat(model=model, messages=messages, options=options)
      content = response.message.content
      self.history.append({"role": "assistant", "content": content})
      yield content
    

  def _handle_tool_calls(self, tool_calls, messages, model, options):
    import subprocess
    yield ""
    for tool_call in tool_calls:
      name = tool_call.function.name
      args = tool_call.function.arguments

      if name == "run_command":
        command = args["command"]
        print(f"\n [BMO] wants to run: {command}")
      elif name == "read_file":
        path = args["path"]
        print(f"\n [BMO] Wants to read: {path}")
      elif name == "search_notes":
        query = args["query"]
        print(f"\n [BMO] wants to search notes for: {query}")

      approval = input(" Approve? (y/n): ").strip().lower()

      if approval != "y":
        tool_result = "Tool call denied by user."
      else:
        if name == "run_command":
          try:
            result = subprocess.run(
              command,
              shell=True,
              capture_output=True,
              text=True,
              timeout=30
            )
            tool_result = result.stdout or result.stderr or "Command ran with no output."
          except subprocess.TimeoutExpired:
            tool_result = "Error: command timed out after 30 seconds."
          except Exception as e:
            tool_result = f"Error running command: {str(e)}"
        elif name == "read_file":
          try:
            with open(path, "r") as f:
              tool_result = f.read()
          except FileNotFoundError:
            tool_result = f"Error: file not found at {path}."
          except Exception as e:
            tool_result = f"Error reading file: {str(e)}"
        elif name == "search_notes":
          if self.rag_store is None:
            tool_result = "RAG not available - vault not configured"
          else:
            results = self.rag_store.search(query)
            if not results:
              tool_result = "No relevant notes found."
            else:
              #format results as readable chunks with source file labels
              parts = []
              for text, source in results:
                parts.append(f"[{source}]\n{text}")
              tool_result = "\n\n---\n\n".join(parts)
      messages.append({"role": "assistant", "tool_calls": [tool_call]})
      messages.append({"role": "tool", "content": tool_result})

    # Model's final response using the tool output 
    full_response = ""
    for chunk in self.client.chat(
      model=model,
      messages=messages,
      stream=True,
      options=options
    ):
      token = chunk.message.content
      if token:
        full_response += token
        yield token
    self.history.append({"role": "assistant", "content": full_response})

  def clear_history(self):
    """ Wipe conversation history """
    self.history = []
