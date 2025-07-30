import os
import hashlib
import pickle
from functools import lru_cache
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load API Key from file
with open("API_KEY.txt", "r") as f:
    GEMINI_API_KEY = f.read().strip()

genai.configure(api_key=GEMINI_API_KEY)

# Global embeddings instance
_embeddings_instance = None

def get_embeddings_instance():
    """Get or create a global embeddings instance for reuse"""
    global _embeddings_instance
    if _embeddings_instance is None:
        print("🔄 Loading HuggingFace embeddings once...")
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance


class RAGPipeline:
    def __init__(self, pdf_path, use_cache=True):
        self.pdf_path = pdf_path
        self.use_cache = use_cache
        self.cache_dir = "rag_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.embeddings = get_embeddings_instance()
        self._initialize_components()

    def _get_cache_path(self, suffix: str) -> str:
        file_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.pkl")

    def _initialize_components(self):
        vectorstore_cache = self._get_cache_path("vectorstore")
        chunks_cache = self._get_cache_path("chunks")

        if self.use_cache and os.path.exists(vectorstore_cache) and os.path.exists(chunks_cache):
            print(f"🔁 Loading cached components for {self.pdf_path}")
            self._load_from_cache()
        else:
            print(f"🆕 Creating components for {self.pdf_path}")
            self._create_components()
            if self.use_cache:
                self._save_to_cache()

        self._initialize_llm_chains()

    def _load_from_cache(self):
        try:
            with open(self._get_cache_path("chunks"), 'rb') as f:
                self.text_chunks = pickle.load(f)

            self.vectorstore = FAISS.load_local(
                self._get_cache_path("vectorstore"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_type="mmr")
            print("✅ Components loaded from cache")

        except Exception as e:
            print(f"⚠️ Cache load failed: {e}. Recreating components.")
            self._create_components()

    def _save_to_cache(self):
        try:
            with open(self._get_cache_path("chunks"), 'wb') as f:
                pickle.dump(self.text_chunks, f)
            self.vectorstore.save_local(self._get_cache_path("vectorstore"))
            print("✅ Components saved to cache")
        except Exception as e:
            print(f"⚠️ Failed to save to cache: {e}")

    def _create_components(self):
        self.documents = PyMuPDFLoader(self.pdf_path).load()
        self.text_chunks = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        ).split_documents(self.documents)

        self.vectorstore = FAISS.from_documents(self.text_chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")

    def _initialize_llm_chains(self):
        self.llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

Guidelines for generating answers:

- Use ONLY the information present in the context provided. Do not use outside knowledge or assumptions.
- Be accurate and formal in tone.
- Include exact details (e.g., number of days/months, specific conditions, monetary values, policy terms, legal references) when present.
- Begin with "Yes" or "No" when relevant, followed by a precise explanation.
- If the information is not available in the context, clearly state: "Information not available in the provided document."
- Limit your answer to ONE single sentence, no matter the complexity.
- Avoid vague language. Prefer concrete facts over generalizations.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        self.qa_llm_chain = LLMChain(llm=self.llm_model, prompt=custom_prompt)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=False,
        )

    @lru_cache(maxsize=500)
    def _cached_ask(self, question_hash: str, question: str) -> str:
        return self.qa_chain.run(question)

    def ask(self, question: str) -> str:
        if self.use_cache:
            question_hash = hashlib.md5(question.encode()).hexdigest()
            return self._cached_ask(question_hash, question)
        else:
            return self.qa_chain.run(question)

    def batch_ask(self, questions: List[str]) -> List[str]:
        return [self.ask(q) for q in questions]

    def clear_cache(self):
        self._cached_ask.cache_clear()

    def get_cache_info(self):
        return {
            "question_cache_info": self._cached_ask.cache_info()._asdict(),
            "vectorstore_cached": os.path.exists(self._get_cache_path("vectorstore")),
            "chunks_cached": os.path.exists(self._get_cache_path("chunks"))
        }

    def preload_similar_questions(self, sample_questions: List[str]):
        print(f"⚡ Preloading {len(sample_questions)} sample questions...")
        for question in sample_questions:
            self.ask(question)
        print("✅ Preloading completed.")
