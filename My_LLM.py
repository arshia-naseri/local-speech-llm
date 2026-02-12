import requests
import json
from ddgs import DDGS


class MyLlm:
    # Public
    SERVER_URL = "http://localhost:11434/api/chat"
    conversation_history: list = None
    instructions: str = None
    Model: str = None
    ModelTemperature = 0.7
    getTokenCount: bool = None
    totalTokensUsed: int = 0
    # Private
    __depth_search__ = 10
    __ONLINE_TRIGGER_LIST__ = [
        "weather",
        "news",
        "today",
        "current",
        "now",
        "latest",
        "price",
    ]  # If see these keywords, search online
    __LLM_CONFIG_INSTRUCTION__ = "Be concise. Answer in 1-3 sentences. Do not give text in latex. Plain text"  # Things I want to set for my llm

    def __init__(
        self,
        instructions: str = None,
        Model: str = "llama3.2",
        getTokenCount: bool = False,
    ):
        self.Model = Model
        self.instructions = instructions if instructions else ""
        self.conversation_history = [
            {
                "role": "system",
                "content": f" {self.instructions}. {self.__LLM_CONFIG_INSTRUCTION__}",
            }
        ]
        self.getTokenCount = getTokenCount

    def __web_search__(self, query, max_results=__depth_search__):
        print("Searching Online ...")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

    def clearChat(self):
        self.conversation_history.clear()
        print("[🧹] Chat Cleared")

    def chat(self, prompt: str, isStream=True, needFullConvo=False, print_output=True):
        # Check if needs any online search
        if self.__needOnlineSearch__(prompt):
            try:
                search_results = self.__web_search__(prompt)
                augmented_prompt = f"""User question: {prompt}
                    Web search results:
                    {search_results}
                    Answer based on the search results above."""
                self.conversation_history.append(
                    {"role": "user", "content": augmented_prompt}
                )
            except Exception:
                print(
                    "[System] No WiFi connection. For updated results, connect to WiFi."
                )
                self.conversation_history.append({"role": "user", "content": prompt})
        else:
            self.conversation_history.append({"role": "user", "content": prompt})

        data = {
            "model": self.Model,
            "messages": self.conversation_history,
            "options": {"num_predict": 512, "temperature": self.ModelTemperature},
            "stream": isStream,
        }

        RES = requests.post(self.SERVER_URL, json=data, stream=isStream)
        response = None

        if print_output:
            print("‣ ", end="")
        if isStream:
            response = self.__chat_with_stream__(RES, print_output)
        else:
            response = self.__chat_without_stream__(RES)

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def __format_tokens__(self, count):
        if count >= 1000:
            return f"{count / 1000:.1f}k"
        return str(count)

    def __print_token_count__(self, prompt_tokens, eval_tokens):
        self.totalTokensUsed = prompt_tokens + eval_tokens
        print(
            f"\n[🤖] Tokens Burned 🔥: {self.__format_tokens__(self.totalTokensUsed)}"
        )

    def __needOnlineSearch__(self, prompt: str):
        return any(word in prompt.lower() for word in self.__ONLINE_TRIGGER_LIST__)

    def __chat_with_stream__(self, RES, print_output=True):
        full_response = ""
        for line in RES.iter_lines():
            if line:
                json_response = json.loads(line)
                if "message" in json_response:
                    chunk = json_response["message"]["content"]
                    full_response += chunk
                    if print_output:
                        print(chunk, end="", flush=True)
                if json_response.get("done") and self.getTokenCount:
                    prompt_tokens = json_response.get("prompt_eval_count", 0)
                    eval_tokens = json_response.get("eval_count", 0)
                    self.__print_token_count__(prompt_tokens, eval_tokens)
        if print_output:
            print("")
        return full_response

    def __chat_without_stream__(self, RES):
        data = RES.json()
        if self.getTokenCount:
            prompt_tokens = data.get("prompt_eval_count", 0)
            eval_tokens = data.get("eval_count", 0)
            self.__print_token_count__(prompt_tokens, eval_tokens)
        return data["message"]["content"]

    # ! Requires string cleaning
    def __str__(self):
        print(f"Chat History w/ Model {self.Model}")
        return str(self.conversation_history)
