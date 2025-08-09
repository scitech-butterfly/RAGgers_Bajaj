from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import io
# Import your functions here
from finalrag import process_query, groq_client

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running. Use /hackrx/run"}

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(payload: QueryRequest):
    url = payload.documents
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return {"error": "Unable to download document"}
        tmp_path = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_path.write(chunk)
        tmp_path.seek(0)

        results = []
        for q in payload.questions:
            response = process_query(q, tmp_path, groq_client)
            results.append({
                "question": q,
                "answer": response
            })

        return {"answers": results}

    except Exception as e:
        return {"error": str(e)}



