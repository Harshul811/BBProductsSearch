from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os 
import pandas as pd 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import getpass


QDRANT_URL="https://89493295-3556-440d-b575-4a17c278da8b.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY ="GyxfvL8cdu1ljvDVKfi5NVPvu7PhFt1U_tuvpqrtUHsiOzIJZ8sQFA"
client = qdrant_client.QdrantClient(QDRANT_URL, QDRANT_API_KEY)
collection_name = "products"

"""
client.create_collection(
    collection_name= collection_name,
    vectors_config= qdrant_client.http.models.VectorParams(
    	size=1536, 
    	distance= qdrant_client.http.models.Distance.COSINE
    	),
)
"""

## 768 for instructor-xl, 1536 as we will be using openAI embeddings 
os.environ["OPENAI_API_KEY"] =  getpass.getpass("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")
embeddings = OpenAIEmbeddings()


# Collection name is thing -  to initialize vector store object
vector_store = Qdrant(
    client=client, 
    collection_name= collection_name, 
    embeddings=embeddings,
)

# Extract product names and descriptions as lists
products_df = pd.read_csv('bigBasketProducts.csv')
product_names = products_df['product'].tolist()
product_descriptions = products_df['description'].tolist()

texts = [] # arrays of text to add to vector store
vector_store.add_texts(product_names[:50])


qa= RetrievalQA.from_chain_type(
    llm= OpenAI(), 
    chain_type="stuff",
    retriever= vector_store.as_retriever()                  
)

query= "How many products are there and what are their names?"
response = qa.run(query)
print(response)

