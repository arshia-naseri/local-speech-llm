import requests
import json
from ddgs import DDGS

class MyLlm:
    # Public
    SERVER_URL = "http://localhost:11434/api/chat"
    conversation_history:list = None
    instructions:str = None
    Model:str = None
    # Private
    __depth_search__ = 10
    __LLM_CONFIG_INSTRUCTION__ = "Be concise. Answer in 1-3 sentences." # Things I want to set for my llm

    def __init__(self, instructions:str = None, Model:str = "llama3.2"):
        self.Model = Model
        self.instructions = instructions
        self.conversation_history = [
            {
                "role": "system",
                "content": self.instructions + " " + self.__LLM_CONFIG_INSTRUCTION__
            }
        ]

    def web_search(self, query, max_results=__depth_search__):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

    def clearChat(self):
        self.conversation_history = None
    
    def __str__(self):
        print("Chat History:")
        print(self.conversation_history)
