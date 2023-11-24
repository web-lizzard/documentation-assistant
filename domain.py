from db.client import client
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma

from langchain.vectorstores.pinecone import Pinecone
from const import INDEX_NAME
import os

from typing import Any


def run_llm_chroma(query: str, chat_history: list[tuple[str, Any]]) -> Any:
    embedding = OpenAIEmbeddings()

    docsearch = Chroma(
        collection_name="my_collection", embedding_function=embedding, client=client
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa({"question": query, "chat_history": chat_history})


def run_llm(query: str, chat_history: list[tuple[str, Any]] = []) -> Any:
    embedding = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embedding)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa({"question": query, "chat_history": chat_history})
