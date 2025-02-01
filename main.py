from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich import print as rich_print
from rich.console import Console
from rich.prompt import Prompt
from rich.padding import Padding


console = Console()


config = dotenv_values(".env")


model = ChatOllama(
    model=config.get("LLM_MODEL", "llama3.2"),
    temperature=0,
)


SYSTEM_MESSAGE_INPUT_STR = """
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
system_message = SystemMessage(SYSTEM_MESSAGE_INPUT_STR)


def select_learning_track(selected_number: int) -> str:
    """
    Pick a learning track based on the selected number.
    Default is 1.
    """
    text_file_name: str = "1.txt"
    match selected_number:
        case 2:
            text_file_name = "2.txt"
        case 3:
            text_file_name = "3.txt"
        case _:
            text_file_name = "1.txt"
    with open(text_file_name, "r", encoding="utf-8") as file:
        return file.read()


def run():
    """ "
    Run Nordic Tales:
    """
    rich_print("[italic green] Welcome Nordic Tales!")
    selected_track_text = select_learning_track(1)
    messages = [system_message, HumanMessage(selected_track_text)]
    streamed_resp = ""
    for chunk in model.stream(messages):
        console.print(chunk.content, end="", style="green", highlight=False, width=85)
        streamed_resp += chunk.content
    messages.append(AIMessage(streamed_resp))
    while True:
        user_input = Prompt.ask("\n Enter your answer:")
        if not user_input:
            console.print("Bye!", style="bold red")
            break
        if user_input == "exit":
            console.print("Bye!", style="bold red")
            break
        messages.append(HumanMessage(user_input))
        console.clear()  # removes the clutter in the console
        streamed_response = ""
        for chunk in model.stream(messages):  # Stream the response
            console.print(chunk.content, end="", style="green", highlight=False)
            streamed_response += chunk.content
        messages.append(AIMessage(streamed_response))

if __name__ == "__main__":
    run()
