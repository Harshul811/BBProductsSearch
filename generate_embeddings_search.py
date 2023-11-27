from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os 
import pandas as pd 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import getpass


QDRANT_URL="https://89493295-3556-440d-b575-4a17c278da8b.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY =""
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
os.environ["OPENAI_API_KEY"] =  getpass.getpass("")
embeddings = OpenAIEmbeddings()


# Collection name is thing -  to initialize vector store object
vector_store = Qdrant(
    client=client, 
    collection_name= collection_name, 
    embeddings=embeddings,
)

# Extract product names and descriptions as lists
products_df = pd.read_csv('bigBasketProducts.csv')

rows_as_text = []
for index, row in products_df.iterrows():
    row_text = ', '.join(f'{column} {value}' for column, value in row.items() if column != "index")
    rows_as_text.append(row_text)
vector_store.add_texts(rows_as_text[:50])  #update to all rows with larger tps limits 


qa= RetrievalQA.from_chain_type(
    llm= OpenAI(), 
    chain_type="stuff",
    retriever= vector_store.as_retriever()                  
)

query= "How many products are there and what are their names?"
response = qa.run(query)
print(response)

