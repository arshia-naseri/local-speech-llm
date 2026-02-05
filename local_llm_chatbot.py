from My_LLM import MyLlm


if __name__ == "__main__":
    myChat = MyLlm()
    
    print("Turn off Wifi & Start Chatting ;) ")
    while True:
        prompt = input("\n>> ")
        if prompt.lower() == '/exit':
            break
        if prompt.lower() == '/clear':
            myChat.clearChat()
            continue
        
        myChat.chat(prompt)

    print("=== LLM CLOSED ===")
    print(myChat)