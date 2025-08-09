from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import io
# Import your functions here
from finalrag import get_index_and_chunks, process_query_with_index, groq_client

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running. Use /hackrx/run"}

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(payload: QueryRequest):
    """
    Downloads the document via streaming into a BytesIO buffer (constant memory usage),
    builds the FAISS index once in a streaming, page-by-page fashion,
    and reuses it for all questions.
    """
    url = payload.documents
    try:
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                return {"error": "Unable to download document"}
            tmp_buffer = io.BytesIO()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp_buffer.write(chunk)
            tmp_buffer.seek(0)

        # Build FAISS index once (streaming mode)
        chunks, index, model = get_index_and_chunks(tmp_buffer)

        results = []
        for q in payload.questions:
            answer = process_query_with_index(q, chunks, index, model, groq_client)
            results.append({
                "question": q,
                "answer": answer
            })

        return {"answers": results}

    except Exception as e:
        return {"error": str(e)}

