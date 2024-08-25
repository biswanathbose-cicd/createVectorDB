import time
from google.api_core.exceptions import ServiceUnavailable

from google.cloud import storage
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

#Import Python modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# from langchain.chains import RetrievalQA
import pickle
import os

MAX_RETRIES = 5

def embed_with_retries(docs, embeddings):
    for attempt in range(MAX_RETRIES):
        try:
            return FAISS.from_documents(docs, embeddings)
        except ServiceUnavailable:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def process_csv(event, context):
    # Initialize the GCS client
    storage_client = storage.Client()

    # Get bucket and file details from the event
    input_bucket_name = os.getenv('INPUT_BUCKET')
    output_bucket_name = os.getenv('OUTPUT_BUCKET')
    file_name = event['name']

    # Get bucket and file details from the event
    bucket_name = event['bucket']
    file_name = event['name']
    #Load the models
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Ensure the file is in the correct folder
    if not file_name.startswith('csv_files/'):
       print(f"File {file_name} is not in the csv_files folder.")
       return


    # Download the CSV file locally
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    local_file_path = f'/tmp/{os.path.basename(file_name)}'
    blob.download_to_filename(local_file_path)

    loader = CSVLoader(file_path=local_file_path, source_column='prompt', encoding='iso-8859-1')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=250,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    vectordb = embed_with_retries(docs, embeddings)
    # vectordb=FAISS.from_documents(docs, embeddings)
    print("VectorDB creation done")
    # Save the vector database locally 
    local_index_path = '/tmp/faiss_index'
    local_documents_path = '/tmp/documents.pkl'
    vectordb.save_local(local_index_path)
    print("save to local index done")


    if os.path.isdir(local_index_path):
        print(f"Contents of {local_index_path}:")
        print(os.listdir(local_index_path))
    else:
        print(f"Directory does not exist: {local_index_path}")

    index_blob = bucket.blob('faiss_index/index.faiss')
    index_pkl_blob = bucket.blob('faiss_index/index.pkl')

    faiss_index_file = os.path.join(local_index_path, 'index.faiss')
    faise_pickle_file = os.path.join(local_index_path, 'index.pkl')

    if os.path.isfile(faiss_index_file):
        index_blob.upload_from_filename(faiss_index_file)
        index_pkl_blob.upload_from_filename(faise_pickle_file)
        print("FAISS index uploaded successfully.")
    else:
        print(f"FAISS index file does not exist: {faiss_index_file}")
 
