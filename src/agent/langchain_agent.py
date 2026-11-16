import os
try:
    from langchain.agents import AgentType, Tool, initialize_agent
except ImportError:
    from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_ollama import ChatOllama

from src.agent.agent_tools import TOOLS

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)

langchain_tools = [
    Tool(
        name="search",
        func=TOOLS["search"],
        description="Поиск релевантной информации в базе документов",
    ),
    Tool(
        name="calc",
        func=TOOLS["calc"],
        description="Математический калькулятор для арифметических выражений",
    ),
    Tool(
        name="get_date",
        func=TOOLS["get_date"],
        description="Возвращает текущую дату в формате ГГГГ-ММ-ДД",
    ),
]

agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
)

if __name__ == "__main__":
    question = "Найди в документах количество дней отпуска и умножь на 8."
    print("Запуск LangChain агента...")
    result = agent.run(question)
    print(f"\nИтоговый ответ: {result}")
