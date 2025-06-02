from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

# Setup memory
memory = ConversationBufferMemory(return_messages=True)

# Setup prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named Gemu."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="put you api key here"
)

# Combine into a runnable chain
from langchain.chains import LLMChain

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Run it
response = chain.invoke({"input": "what is my Name"})
print(response["text"])
