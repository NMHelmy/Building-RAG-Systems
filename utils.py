import os
from PyPDF2 import PdfReader
from docx import Document

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages])

def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            docs.append(load_txt(path))
        elif filename.endswith('.pdf'):
            docs.append(load_pdf(path))
        elif filename.endswith('.docx'):
            docs.append(load_docx(path))
    return docs

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
