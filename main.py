from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


model = ChatOllama(
    model="llama3.2",
    temperature=0,
)


messages = [
    SystemMessage("Translate the following from English into Norwegian"),
    HumanMessage("Hello, how are you?"),
]

model.invoke(messages)
