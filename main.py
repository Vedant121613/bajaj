from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from utils import download_pdf
import traceback

app = FastAPI()

# Request payload structure
class RequestPayload(BaseModel):
    documents: str  # URL to the PDF
    questions: list[str]  # List of questions to be answered

@app.api_route("/", methods=["GET", "HEAD"])
def root(request: Request):
    return {"status": "API running"}

@app.post("/hackrx/run")
async def run_rag(payload: RequestPayload):
    """
    Process a PDF and answer questions using the RAG pipeline.
    """
    try:
        # Download PDF from URL
        pdf_path = await download_pdf(payload.documents)

        # Initialize RAG pipeline and get answers
        rag = RAGPipeline(pdf_path)
        answers = rag.batch_ask(payload.questions)

        return {"answers": answers}

    except HTTPException as he:
        raise he

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in run_rag: {traceback_str}")
        return JSONResponse(
            status_code=502,
            content={"error": str(e), "trace": traceback_str}
        )
