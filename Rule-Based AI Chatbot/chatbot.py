from datetime import datetime

print("Welcome to DecodeBot!")
print("Type 'bye' to exit.")

while True:

    user = input("You: ").lower().strip()

    if user in ["hi", "hello", "hey"]:
        print("Bot: Hello! Nice to meet you.")

    elif user == "how are you":
        print("Bot: I am doing great!")

    elif user == "what is your name":
        print("Bot: My name is DecodeBot.")

    elif user == "who created you":
        print("Bot: I was created as a Rule-Based AI Chatbot project.")

    elif user == "date":
        print("Bot:", datetime.now().strftime("%d-%m-%Y"))

    elif user == "time":
        print("Bot:", datetime.now().strftime("%H:%M:%S"))

    elif user in ["thanks", "thank you"]:
        print("Bot: You're welcome!")

    elif user in ["bye", "exit", "quit"]:
        print("Bot: Goodbye!")
        break

    else:
        print("Bot: Sorry, I don't understand.")