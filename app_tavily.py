from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_openai_tools_agent, AgentExecutor  
from langchain.tools.retriever import create_retriever_tool           
from langchain_community.tools.tavily_search import TavilySearchResults  
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import dotenv_values
from sqlalchemy import create_engine
from datetime import datetime
import os
import uvicorn

config = dotenv_values(".env")
os.environ['TAVILY_API_KEY'] = config.get("TAVILY_API_KEY")

llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    temperature=0.7,
    top_p=0.92,
)

embedding_llm = AzureOpenAIEmbeddings(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
)

client = QdrantClient(url=config.get("QDRANT_URL"), api_key=config.get("QDRANT_API_KEY"))
qdrant = QdrantVectorStore(
    client=client,
    collection_name="Mother_HandBook",
    embedding=embedding_llm
)

retriever_tool = create_retriever_tool(
    qdrant.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5}),
    "breastfeeding_assistant",
    "當問題與母乳哺育、育兒相關時，請使用這個工具來回答。"
)

tavily_tool = TavilySearchResults(max_results=3, recent_only=True)

tools = [retriever_tool, tavily_tool]

current_date = datetime.now().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    你是一位熱情且友善的母乳哺育助手，幫助使用者解決母乳餵養問題。
    你可以使用以下工具：
    - breastfeeding_assistant: 查詢母乳哺育和育兒相關知識
    - tavily_search_results_json: 查詢即時資訊，例如新聞、醫院、天氣等

    今天日期是 {current_date}。
    如果是母乳哺育問題，請優先使用 breastfeeding_assistant。
    如果問題涉及即時資訊（例如新聞、天氣），請使用 tavily_search_results_json。
    """),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

def get_session_history(session_id):
    engine = create_engine("sqlite:///../tavily.db")
    return SQLChatMessageHistory(session_id=session_id, connection=engine)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "anonymous_user"

@app.post("/query")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        user_input = request.message

        print(f"🔍 接收到請求: session_id={session_id}, message={user_input}")

        session_history = get_session_history(session_id)
        chat_history = [{"role": "user", "content": msg.content} if msg.type == "human"
                        else {"role": "ai", "content": msg.content}
                        for msg in session_history.messages]

        agent = create_openai_tools_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

        search_results = tavily_tool.run(user_input)

        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "search_results": search_results,
        }, config={"configurable": {"session_id": session_id}})

        session_history.add_user_message(user_input)
        session_history.add_ai_message(result["output"])

        return {"answer": result["output"]}

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"⚠️ FastAPI 內部錯誤: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
