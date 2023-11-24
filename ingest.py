from db.client import client
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

from const import INDEX_NAME


def ingest_document_chroma():
    loader = DirectoryLoader(
        path="tmp",
        glob="**/*.html",
        show_progress=True,
        use_multithreading=True,
        loader_cls=BSHTMLLoader,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", "", " "]
    )

    raw_docs = loader.load()

    print("document before split", raw_docs)

    documents = splitter.split_documents(documents=raw_docs)

    for doc in documents:
        old_path = doc.metadata["source"]
        new_path = old_path.replace("tmp/", "https:/")
        doc.metadata.update({"source": new_path})

    print("document after spli", documents)
    print(len(documents))

    embedding = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        client=client,
        collection_name="my_collection",
    )


def ingest_doc():
    loader = DirectoryLoader(
        recursive=True,
        path="tmp",
        glob="**/*.html",
        show_progress=True,
        loader_cls=BSHTMLLoader,
        use_multithreading=True,
    )
    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", "", " "]
    )

    documents = text_splitter.split_documents(documents=raw_document)

    for doc in documents:
        old_path = doc.metadata["source"]
        new_path = old_path.replace("tmp/developer.infermedica.com/", "https:/")
        doc.metadata.update({"source": new_path})

    embedding = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embedding, index_name=INDEX_NAME
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import pinecone
    import os

    load_dotenv()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
    )

    ingest_document_chroma()
