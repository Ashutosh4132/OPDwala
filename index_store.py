from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import ServerlessSpec
from pinecone import Pinecone
from typing import List
import torch



from dotenv import load_dotenv
import os 
load_dotenv()

# LOAD ALL PDF FILES TO UPLOAD
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls = PyPDFLoader
    )
    
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    
    minimal_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source" : src}
            )
        )
    return minimal_docs




def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts = text_splitter.split_documents(minimal_docs)
    return texts



def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-V2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = {"device" : "cuda" if torch.cuda.is_available() else "cpu" }
    )
    return embeddings




extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts = text_split(minimal_docs)
embedding = download_embeddings()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key = pinecone_api_key)



index_name = "opdwaala"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)



docsearch = PineconeVectorStore.from_documents(
    documents=texts,
    embedding=embedding,
    index_name = index_name
)
retriever = docsearch.as_retriever(search_type = "similarity" , search_kwargs= {"k":3})