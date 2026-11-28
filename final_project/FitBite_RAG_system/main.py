
from app.agent import NutritionAgent


def main():
    agent = NutritionAgent()

    print("\nNutrition Agent is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = agent.chat(question)
            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
