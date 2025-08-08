from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import tempfile
import requests

# Import your functions here
from RAGgers.pipeline import process_query, groq_client

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(payload: QueryRequest):
    url = payload.documents
    try:
        # Download PDF to temp file
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": "Unable to download document"}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

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
