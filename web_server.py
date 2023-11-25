from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from domain import run_llm_chroma, GPT_MODEL
from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    body: str
    model: GPT_MODEL | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/query", response_model=QueryResponse)
async def query_assistant(query: Query):
    response = run_llm_chroma(
        query=query.body, model=query.model if query.model else GPT_MODEL.GPT_3
    )

    sources = set(
        [source.metadata["source"] for source in response["source_documents"]]
    )

    return QueryResponse(answer=response["answer"], sources=list(sources))


@app.websocket("/chat")
async def chat_with_model(websocket: WebSocket):
    await websocket.accept()
    chat_history = []

    while True:
        query = Query.model_validate(await websocket.receive_json())
        response = run_llm_chroma(
            query=query.body,
            model=query.model if query.model else GPT_MODEL.GPT_3,
            chat_history=chat_history,
        )

        await websocket.send_text(response["answer"])
        chat_history.append((query, response["answer"]))


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8001/chat");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(JSON.stringify({body: input.value}))
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get_index():
    return HTMLResponse(html)
