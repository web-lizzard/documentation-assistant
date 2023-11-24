if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn
    import pinecone
    from web_server import app
    import os

    load_dotenv()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
    )

    uvicorn.run(app, host="0.0.0.0", port=8001)
