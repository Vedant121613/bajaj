from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()



# # Load API Key from file
# with open("API_KEY.txt", "r") as f:
#     GEMINI_API_KEY = f.read().strip()
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

class RAGPipeline:
    def __init__(self, pdf_path):
        self.documents = PyMuPDFLoader(pdf_path).load()
        self.text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(self.documents)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.text_chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # You can use gemini-2.0-flash for faster
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )

#         template = """
# You are an expert document analyst specializing in insurance, legal, and compliance documents. You will answer questions using ONLY the provided context from the document retrieval system.

# RETRIEVED CONTEXT:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# 1. Answer the question using ONLY information from the retrieved context above
# 2. Be precise and factual - include specific numbers, dates, percentages, and conditions when mentioned
# 3. If the context contains relevant information, provide a comprehensive answer
# 4. If the information is insufficient or not available in the context, clearly state "Information not available in the provided document"
# 5. Structure your answer clearly and logically
# 6. Reference specific context sections when making claims (e.g., "According to Context 1...")
# 7. Keep your response focused and under 300 words

# ANSWER:
# """
#         custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an expert document analyst specializing in insurance, legal, and compliance documents. You will answer questions using ONLY the provided context from the document retrieval system.

# RETRIEVED CONTEXT:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# 1. Answer the question using ONLY information from the retrieved context above
# 2. Be precise and factual - include specific numbers, dates, percentages, and conditions when mentioned
# 3. If the context contains relevant information, provide a comprehensive answer
# 4. If the information is insufficient or not available in the context, clearly state "Information not available in the provided document"
# 5. Structure your answer clearly and logically
# 6. Reference specific context sections when making claims (e.g., "According to Context 1...")
# 7. Keep your response focused and under 300 words

# ANSWER:
# """
# )       
        custom_prompt = PromptTemplate(
input_variables=["context", "question"],
template="""
You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

Guidelines for generating answers:

    Use ONLY the information present in the context provided. Do not use outside knowledge or assumptions.

    Be accurate, concise, and formal in tone.

    Include exact details (e.g., number of days/months, specific conditions, monetary values, policy terms, legal references) when present.

    When relevant, begin with "Yes" or "No" followed by a precise explanation.

    If the information is not available in the context, clearly state: "Information not available in the provided document."

    Limit your answer to 3–5 sentences unless the content requires additional explanation.

    Avoid vague language. Prefer concrete facts over generalizations.

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
        # prompt = ChatPromptTemplate.from_template(template)
        # output_parser = StrOutputParser()

        # self.rag_chain = (
        #     {"context": self.retriever, "question": RunnablePassthrough()}
        #     | prompt
        #     | self.llm_model
        #     | output_parser
        # )

    def ask(self, question):
        return self.qa_chain.run(question)
