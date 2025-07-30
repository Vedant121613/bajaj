from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from utils import download_pdf
from authorization import validate_token
import asyncio
import hashlib
from functools import lru_cache
from typing import Dict, List
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import os
import traceback

app = FastAPI()

rag_cache: Dict[str, RAGPipeline] = {}
pdf_cache: Dict[str, str] = {}
executor = ThreadPoolExecutor(max_workers=4)

class RequestPayload(BaseModel):
    documents: str
    questions: list[str]

def get_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=100)
def cached_validate_token(token: str) -> bool:
    return validate_token(token)

async def download_pdf_async(url: str) -> str:
    cache_key = get_cache_key(url)
    if cache_key in pdf_cache:
        print(f"Using cached PDF for: {url}")
        return pdf_cache[cache_key]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    filename = f"temp_{cache_key}.pdf"
                    async with aiofiles.open(filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    pdf_cache[cache_key] = filename
                    print(f"Downloaded and cached: {filename}")
                    return filename
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status}")
    except Exception as e:
        print(f"Async download failed, fallback to sync: {e}")
        try:
            loop = asyncio.get_event_loop()
            pdf_path = await loop.run_in_executor(executor, download_pdf, url)
            pdf_cache[cache_key] = pdf_path
            return pdf_path
        except Exception as sync_e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {sync_e}")

def get_or_create_rag(pdf_path: str) -> RAGPipeline:
    if pdf_path not in rag_cache:
        print(f"Creating new RAG for {pdf_path}")
        rag_cache[pdf_path] = RAGPipeline(pdf_path)
    else:
        print(f"Using cached RAG for {pdf_path}")
    return rag_cache[pdf_path]

async def process_question_async(rag: RAGPipeline, question: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.ask, question)

async def process_questions_batch(rag: RAGPipeline, questions: List[str]) -> List[str]:
    if hasattr(rag, 'batch_ask'):
        print("Using batch_ask method")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, rag.batch_ask, questions)
    else:
        print("Using async ask method")
        tasks = [process_question_async(rag, q) for q in questions]
        return await asyncio.gather(*tasks)


@app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root(request: Request):
    return {"message": f"RAG Pipeline API is running with method {request.method}"}


@app.post("/hackrx/run")
async def run_api(payload: RequestPayload, authorization: str = Header(None)):
    try:
        print(f"Authorization: {authorization}")
        if not cached_validate_token(authorization):
            raise HTTPException(status_code=401, detail="Invalid or missing token")

        print(f"Processing: {payload.documents}")
        pdf_path = await download_pdf_async(payload.documents)
        rag = get_or_create_rag(pdf_path)
        answers = await process_questions_batch(rag, payload.questions)
        return {"answers": answers}

    except Exception as e:
        print("❌ EXCEPTION OCCURRED:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/preload")
async def preload_document(url: str, authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    try:
        pdf_path = await download_pdf_async(url)
        get_or_create_rag(pdf_path)
        return {"message": "Document preloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hackrx/cache/status")
async def cache_status(authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return {
        "cached_documents": len(rag_cache),
        "cached_pdfs": len(pdf_cache)
    }

@app.delete("/hackrx/cache")
async def clear_cache(authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    for pdf_path in pdf_cache.values():
        if os.path.exists(pdf_path) and pdf_path.startswith("temp_"):
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"Error deleting {pdf_path}: {e}")
    rag_cache.clear()
    pdf_cache.clear()
    cached_validate_token.cache_clear()
    return {"message": "Cache cleared successfully"}

@app.get("/hackrx/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
    for pdf_path in pdf_cache.values():
        if os.path.exists(pdf_path) and pdf_path.startswith("temp_"):
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"Cleanup error for {pdf_path}: {e}")
    executor.shutdown(wait=True)
    print("Shutdown cleanup complete")