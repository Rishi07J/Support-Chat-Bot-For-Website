from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader

def get_website_data(sitemap_url):
    loader = SitemapLoader(sitemap_url)
    docs = loader.load()  # This is synchronous
    return docs



def split_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks


def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone as PineconeClient

def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    # Initialize Pinecone client
    pc = PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

    # Create the index manually if it doesn't exist (optional)
    if pinecone_index_name not in [i['name'] for i in pc.list_indexes()]:
        pc.create_index(name=pinecone_index_name, dimension=384, metric="cosine")  # 384 is dimension for "all-MiniLM-L6-v2"

    # Push documents
    index = PineconeStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return index





def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    # Initialize Pinecone client
    PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)
    index = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
    return index


def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs
