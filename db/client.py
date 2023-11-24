import chromadb
from chromadb.config import Settings


client = chromadb.HttpClient(settings=Settings(allow_reset=True))
