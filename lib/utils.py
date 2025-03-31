import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _get_byte_size(bytes:int, measure_type:str='mb', debug:bool=False)->None:
    """Calculates the size of bytes to desire data type.

    Args:
        bytes (int): Numeric bytes.
        measure_type (str, optional): Type of byte conversion. Defaults to 'mb'.
        debug (bool, optional): Controls printing of byte conversion. Defaults to False.
    """
    file_size_kb = bytes / 1024
    file_size_mb = file_size_kb / 1024
    file_size_gb = file_size_mb / 1024

    if measure_type.strip().lower()=='mb':
        file_size = file_size_mb
    elif  measure_type.strip().lower()=='gb':
        file_size = file_size_gb
    elif  measure_type.strip().lower()=='kb':
        file_size = file_size_kb
    else:
        print('Not Implemented measure_type should be kb, mb, gb.')
    if debug:
        print(f'{file_size:,.2f} {measure_type.upper()}')

def load_document(file_path:str, encoding:str='utf8'):
    """Load document base on file type.

    Args:
        file_path (str): Path to file.
        encoding (str, optional): Data encoding. Defaults to 'utf8'.

    Raises:
        Exception: File extension not implemented.

    Returns:
        list: Document list.
    """
    bytes =os.stat(file_path).st_size
    _get_byte_size(bytes=bytes)

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        document = loader.load()

    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        document = loader.load()
    
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path,encoding=encoding)
        document = loader.load()
    else:
        raise Exception("File Not Implemented")
    
    return document

def create_vector_store(doc_path:str, model:str,
                        chunk_size:int=1_200,chunk_overlap:int=200)->InMemoryVectorStore:
    """Creates a in memory vector store.

    Args:
        doc_path (str): Path to data.
        model (str): LLM model name.
        chunk_size (int, optional): Chunk size. Defaults to 1_200.
        chunk_overlap (int, optional): Overlap chunck size. Defaults to 200.

    Returns:
        InMemoryVectorStore: Vector db.
    """
    docs = load_document(file_path=doc_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    try:
        del vectorstore
    except:
        pass
    vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OllamaEmbeddings(model=model))
    return vectorstore