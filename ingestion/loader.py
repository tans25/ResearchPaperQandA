from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
import re 
from pypdf import PdfReader
from langchain_core.documents import Document

def remove_irrelevant_pieces(docs):
    cutoff_index = len(docs)
    for i, doc in enumerate(docs):
        text = doc.page_content.lower()
        patterns = [
            r'\breferences\b\s*\n',
            r'\bbibliography\b\s*\n',
            r'\bworks cited\b',
            r'\breferences\b\s*$',
            r'\bappendix\b\s*[a-z]?\s*[\n:]',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                cutoff_index = i
                break
        
        if cutoff_index < len(docs):
            break
    removed = len(docs) - cutoff_index
    if removed > 0 :
        print(f"Removed {removed} chunks.")
    return docs[:cutoff_index]


def load_pdfs(uploaded_files):
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for filename in uploaded_files:
        print("here")
        # filepath = os.path.join(pdf_dir, filename)
        reader = PdfReader(filename)
        docs=[]
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(
                page_content=text,
                metadata={"source": filename.name, "page": i}
            ))
        docs = remove_irrelevant_pieces(docs)
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)
    return all_splits

