import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import dotenv_values


config = dotenv_values(".env")


model = ChatOllama(
    model=config.get("LLM_MODEL", "llama3.2"),
    temperature=0,
)


system_message = SystemMessage("Translate the following from English into Norwegian")


def run():
    print("Welcome Nordic Tales!")
    for user_input in sys.stdin:
        user_input = user_input.strip()
        if not user_input:
            break
        if user_input == "exit":
            break
        messages = [system_message, HumanMessage(user_input)]
        resp = model.invoke(messages)
        print(resp.content)

if __name__ == "__main__":
    run()
