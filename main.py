from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from utils import download_pdf
from authorization import validate_token  # ✅ Import your function

app = FastAPI()

class RequestPayload(BaseModel):
    documents: str  # URL
    questions: list[str]

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/hackrx/run")
async def run_api(payload: RequestPayload, authorization: str = Header(None)):
    if not validate_token(authorization):  # ✅ Use your custom validator
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    try:
        pdf_path = download_pdf(payload.documents)
        rag = RAGPipeline(pdf_path)
        answers = [rag.ask(q) for q in payload.questions]
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
