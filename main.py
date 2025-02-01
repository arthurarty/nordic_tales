import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import dotenv_values


config = dotenv_values(".env")


model = ChatOllama(
    model=config.get("LLM_MODEL", "llama3.2"),
    temperature=0,
)


system_message_input = """
You are a professional language teacher tasked with conducting a lesson based on a provided script. Your goal is to teach the student certain concepts while following the script closely.

Now, follow these guidelines for teaching:

1. Follow the lesson script closely, covering all the necessary information and concepts.
2. Ask questions and engage the student in the lesson. Encourage active participation.
3. Keep your explanations simple, concise, and easy to understand.
4. Use examples when appropriate to illustrate concepts.
5. Provide pronunciations where necessary.
6. Provide positive reinforcement and encouragement throughout the lesson.

Remember to stay in character as a professional, friendly, and patient language teacher throughout the interaction. Do not deviate from the lesson script or introduce topics not covered in it. Break down the lesson into manageable parts. DO NOT overwhelm the student with too much information at once.

Conduct the lesson as a chat between you (the teacher) and the student. Include ALL the information. Keep the messages short and concise.
"""
system_message = SystemMessage(system_message_input)


def select_learning_track(selected_number: int) -> str:
    text_file_name: str = "1.txt"
    match selected_number:
        case 2:
            text_file_name = "2.txt"
        case 3:
            text_file_name = "3.txt"
        case _:
            text_file_name = "1.txt"
    with open(text_file_name, 'r', encoding='utf-8') as file:
        return file.read()


def run():
    print("Welcome Nordic Tales!")
    selected_track_text = select_learning_track(1)
    messages = [system_message, HumanMessage(selected_track_text)]
    resp = model.invoke(messages)
    messages.append(AIMessage(resp.content))
    print(resp.content)
    for user_input in sys.stdin:
        user_input = user_input.strip()
        if not user_input:
            break
        if user_input == "exit":
            break
        messages.append(HumanMessage(user_input))
        resp = model.invoke(messages)
        messages.append(AIMessage(resp.content))
        print(resp.content)

if __name__ == "__main__":
    run()
