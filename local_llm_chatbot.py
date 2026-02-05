from My_LLM import MyLlm


if __name__ == "__main__":
    beaverChat = MyLlm()
    
    while True:
        prompt = input(">> ")
        if prompt.lower() == '/exit':
            break
        
        beaverChat.chat(prompt)
    
    print("=== LLM CLOSED ===")