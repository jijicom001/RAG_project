from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import dotenv_values
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel
from sqlalchemy import create_engine
from langchain_community.tools import TavilySearchResults
from langchain.memory import ConversationBufferWindowMemory
import json
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.llms import ollama 
from langchain_ollama import ChatOllama


config= dotenv_values(".env")


llm = ChatOllama(
    model="llama3.2:latest",
     base_url="http://localhost:11434",
    temperature=0.7,
    top_p=0.92,
)

embedding_llm= AzureOpenAIEmbeddings(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
)


#-------- 第一次把文稿放入Qdrant 資料庫-------------

# loader= PyPDFLoader("../docs/AI.pdf")
# pages= loader.load_and_split()

# text_splitter= RecursiveCharacterTextSplitter(
#     separators=["\n"],
#     chunk_size= 1000,
#     chunk_overlap=200,

# )
# splits= text_splitter.split_documents(pages)

# loader =Docx2txtLoader("../docs/母乳哺餵問答.docx")
# data =loader.load()
# text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs =text_splitter.split_documents(data)

# qdrant= QdrantVectorStore.from_documents(
#     splits,
#     embedding=embedding_llm,
#     url=config.get("QDRANT_URL"),
#     api_key=config.get("QDRANT_API_KEY"),
#     collection_name="Mother_HandBook",
# )

app= FastAPI()
class ChatRequest(BaseModel):
    message: str
    
client = QdrantClient(url=config.get("QDRANT_URL"),api_key=config.get("QDRANT_API_KEY"))
qdrant = QdrantVectorStore(
        client= client,
        collection_name="Mother_HandBook",
        embedding=embedding_llm
    )
tools = [TavilySearchResults(top_k=5)]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位熱情且專業的母乳哺育與育兒專家。
    你的回答應該自然且易懂，並根據對話歷史提供幫助。
    當有額外資訊可供參考時，請自然地將其整合進回答中，不要提及「根據參考資料」或「提供的資訊」等字眼。
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

    

session_memory = {}

def get_session_memory(session_id):

    if session_id not in session_memory:
        session_memory[session_id] = ConversationBufferWindowMemory(
            memory_key="history",
            return_messages=True,
            k=3  
        )
    return session_memory[session_id]

@app.post("/query")
async def chat(request: Request):
    try:
        body = await request.json()

        session_id = body.get("user_id") or body.get("session_id") or "anonymous_user"
        user_input = body.get("message", "")

        print(f" FastAPI 接收到請求: session_id={session_id}, message={user_input}")

        memory = get_session_memory(str(session_id))
        chat_history = memory.load_memory_variables({})["history"]

        retriever = qdrant.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})

        qa_chain = (
            {
                "context": lambda x: retriever.invoke(x.get("input", "")),
                "history": lambda _: chat_history,
                "input": lambda x: x.get("input", ""),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = qa_chain.invoke({"input": user_input})

        memory.save_context({"input": user_input}, {"answer": answer})

        print(f"FastAPI 生成回應: {answer}")

        return {"answer": answer} 

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"FastAPI 內部錯誤: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# response = chain_with_history.invoke(
#     {"input": "請問你是誰？請簡短回答"},
#     config=config  
# )
# print("AI:", response["answer"])

# while True:
#     response = chain_with_history.invoke(
#     {"input": input()},
#     config=config  
#     )   
#     print("AI:", response["answer"])