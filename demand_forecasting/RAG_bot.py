# Databricks notebook source
# MAGIC %md
# MAGIC #Confirm Mount to Azure Storage Account

# COMMAND ----------

# dbutils.fs.mounts()

# COMMAND ----------

# TEST CELL
# Ths cell just reads a txt doc from the container of interest within the storage account
file_path = "/mnt/Milos_Storage_Container/"
file_name = "TEST DOC.txt"

df = spark.read.text(file_path + file_name)
df.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC #Installs, Functions, and Secrets

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installs/Imports

# COMMAND ----------

!pip install -qU PyPDF2 langchain==0.0.292 openai==0.28.0 datasets==2.10.1 pinecone-client==2.2.4 tiktoken==0.5.1 matplotlib tqdm autopep8

import io
import os
import time
import pandas as pd
import numpy
import seaborn
from pyspark.sql import SparkSession
import PyPDF2
import requests
import langchain
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from datasets import load_dataset
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Utility Functions

# COMMAND ----------

def create_secret_scope():
    payload = {
        "scope": "AzureOpenAI_APIs",
        "initial_manage_principal": "users",
        "scope_backend_type": "DATABRICKS",
    }
    baseURL = "https://adb-239432.12.azuredatabricks.net"
    my_token = dbutils.secrets.get(scope="key-vault-secret", key="databricks-token")
    postheaders = {"Authorization": "Bearer {}".format(my_token)}
    endpoint = "/api/2.0/secrets/scopes/create"
    URL = baseURL + endpoint
    response = requests.post(URL, headers=postheaders, json=payload)
    # if response.ok:
    #     responseData = response.json()
    #     jobsData = responseData['secrets']
    # else:
    #     jobsData = None
    #     print('No access on jobs:')
    # return jobsData
    print(response.status_code)


def get_secret_scopes():
    baseURL = "https://adb-212.azuredatabricks.net"
    my_token = dbutils.secrets.get(scope="key-vault-secret", key="databricks-token")
    postheaders = {"Authorization": "Bearer {}".format(my_token)}
    endpoint = "secrets/scopes/list"
    URL = baseURL + endpoint
    response = requests.get(URL, headers=postheaders)
    if response.ok:
        responseData = response.json()
        jobsData = responseData["scopes"]
    else:
        jobsData = None
        print("No access on jobs:")
    return jobsData


# The 'MilosAI Bot' secret must be deleted because it is my (Scott Grove) personal API key for OPENAI!
def add_secret():
    payload = {
        "scope": "AzureOpenAI_APIs",
        "key": "MilosAI Bot",
        # "string_value": "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",
    }
    baseURL = "https://adb-23942.12.azuredatabricks.net"
    my_token = dbutils.secrets.get(scope="key-vault-secret", key="databricks-token")
    postheaders = {"Authorization": "Bearer {}".format(my_token)}
    endpoint = "/api/secrets/put"
    URL = baseURL + endpoint
    response = requests.post(URL, headers=postheaders, json=payload)
    # if response.ok:
    #     responseData = response.json()
    #     jobsData = responseData['secrets']
    # else:
    #     jobsData = None
    #     print('No access on jobs:')
    # return jobsData
    print(response.status_code)


def get_secrets(scope_name):
    parameters = {"scope": {scope_name}}
    baseURL = "https://adb-23959472.12.azuredatabricks.net"
    my_token = dbutils.secrets.get(scope="key-vault-secret", key="databricks-token")
    postheaders = {"Authorization": "Bearer {}".format(my_token)}
    endpoint = "/api/2.0/secrets/list"
    URL = baseURL + endpoint
    response = requests.get(URL, headers=postheaders, params=parameters)
    if response.ok:
        responseData = response.json()
        jobsData = responseData["secrets"]
    else:
        jobsData = None
        print("No access on jobs:")
    return jobsData

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##Project-Specific Functions

# COMMAND ----------

# This function reads the PDFs, extracts the text, and creates tuples attaching metadata to the pages of text. As currently written, the tuples are triples consisting of (1) the container & file name (together), (2) the page number, and (3) the text from the page.
def read_pdf(binary_pdf):
    # Convert bytes to a file-like object
    pdf_stream = io.BytesIO(binary_pdf[1])

    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    num_pages = len(pdf_reader.pages)

    # Initialize an empty list
    triples = []

    for i in range(num_pages):
        # Extract text from each page
        page_text = pdf_reader.pages[i].extract_text()
        page_number = i + 1
        # Create a triple with the document name, page number, and text content
        triple = (binary_pdf[0], page_number, page_text)

        # Append the triple to the list
        triples.append(triple)

    return triples


# This length function tells us how many tokens a chunk of text represents.
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)    

# This function performs a similarity search of the knowledge base and attaches the top results to the query.
def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # retrieve and consolidate the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Secret Scope and Secrets

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** Should the creation of Scopes and Secrets be done in a separate notebook?
# MAGIC \
# MAGIC \
# MAGIC **Note:** The 'MilosAI Bot' secret must be deleted because it is my (Scott Grove) personal API key for OPENAI.
# MAGIC     It is only being temporarily used (with permission) to troubleshoot/run an unrelated query. There is a "backstop" in the code (just after the 'Rag Chatbot Example') to keep any Milo's Data from being used with it when using the "Run All Below" functionality. 

# COMMAND ----------

# MAGIC %md
# MAGIC **The details of the functions in this section can be found above, in the 'Utility Functions' section.**
# MAGIC
# MAGIC 1. Create a Secret Scope for the necessary Secrets. 
# MAGIC
# MAGIC 2. Create Databricks Secret for OpenAI API Access Key.
# MAGIC
# MAGIC 3. Create Databricks Secret for Pinecone Access Key. Initially, this will be added to the same Secret Scope containing the OpenAI Access Key Secret ("2." above).
# MAGIC     Figure out the norms around Secret Scope usage; the current scope needs to be
# MAGIC     renamed (deleted and recreated) for the sake of accuracy, prior to creating and adding the secret for the Pinecone API Access Key.
# MAGIC     The other secrets contained in the scope will need to be recreated (I think there is only one at the moment).

# COMMAND ----------

# These are defined in 'Utility Functions' (above).
# create_secret_scope()
# add_secret()

# Confirm
get_secret_scopes()
# get_secrets('AzureOpenAI_APIs')

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Spark Session, Read PDFs, Extract Text, and Attach Metadata

# COMMAND ----------

# Initialize Spark Session
spark = SparkSession.builder.master("local[*]").appName("SparkApp").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC >The first line in the following cell reads ALL pdf files from the mounted Azure storage container. This is convenient, but could be a bad thing. ONLY USE AS WRITTEN IF YOU ARE SURE YOU WANT TO READ ALL OF THE PDFS IN THE SPECIFIED CONTAINER.

# COMMAND ----------

binary_pdfs_rdd = spark.sparkContext.binaryFiles("/mnt/MilosAI_Storage_Container/*.pdf")

# Use flatMap transformation to read each PDF and extract text at the page level.
# flatMap will flatten the list of lists into a single RDD of tuples
text_contents_rdd = binary_pdfs_rdd.flatMap(read_pdf)

# Collect results or continue further processing
text_contents = text_contents_rdd.collect()

# At this point, text_contents is a list of triples
# Each triple contains the filename, the page number, and the extracted text from that page of the PDF.

# COMMAND ----------

# MAGIC %md
# MAGIC >What is an RDD? https://www.databricks.com/glossary/what-is-rdd
# MAGIC

# COMMAND ----------

# binary_pdfs_rdd.collect()

# COMMAND ----------

print(f"length of 'binary_pdfs_rdd' = {len(binary_pdfs_rdd.collect())}")
# If I understand, I think the length of 'binary_pdfs_rdd' corresponds to the number of pdf documents in the container.
print(f"length (in pages) of 'text_contents' = {len(text_contents)}")
# The definition of the 'read_pdf' function implies the length of 'text_contents' should correpond to the number of pages in the PDFs.

# COMMAND ----------

# print(text_contents)
# This should contain a tuple per page of text.

# COMMAND ----------

# Filtering the extremely token-inefficient pages with the title and the table of contents out:
filtered_text_contents = text_contents[4:]
print(f"length of 'filtered_text_contents' = {len(filtered_text_contents)}")

# COMMAND ----------

# check
# filtered_text_contents[0]

# COMMAND ----------

# check
# filtered_text_contents[0][2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Tokenizer, Count Tokens, and Chunk Text

# COMMAND ----------

# Tokenizers are specific to the LLM being used because tokens are defined differently from LLM to LLM.

tokenizer = tiktoken.get_encoding("cl100k_base")
# The following may a be a more robust way to choose the correct encoder:
# tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

# COMMAND ----------

# MAGIC %md
# MAGIC >For the tokenizer (above), we defined the encoder as "cl100k_base". This is a specific tiktoken encoder used by gpt-3.5-turbo.

# COMMAND ----------

# create a list to hold the token counts for each tuple:
token_counts = []
# count the tokens in the text element of each tuple:
for page_tuple in filtered_text_contents:
    token_counts.append(tiktoken_len(page_tuple[2]))
print(token_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC >The text will be split into chunks one page at a time, therefore none of the context on a given page will be included in a chunk containing text from another page. This is a limitation/weakness of this method, but can help us learn about more effective text prep (e.g., avoid sentences/paragraphs that cross page breaks as much as possible).
# MAGIC
# MAGIC >It seems like this would be less of an issue if we were pulling in text from webpages; they don't generally have as many page breaks relative to the volume of text. For this same reason, it likely wouldn't be an issue here either if we were creating our tuples at the document level. But, we wanted to attach the page number to the text (it's in the text but isn't labeled as the page number), so we created the tuples at the page level.

# COMMAND ----------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # maximum length in characters
    chunk_overlap=20,  # number of characters overlap between chunks
    length_function=len,
    separators=["\n\n\n", "\n\n", "\n", " ", ""],
)

# COMMAND ----------

list_of_chunks = []
for page_tuple in filtered_text_contents:
    list_of_chunks.append(text_splitter.split_text(page_tuple[2]))
# Note that 'list_of_chunks' is a list of lists, the chunks are still grouped by page.


# print(list_of_chunks[0][1])
# print()
# print(f'\'list_of_chunks\':')
# print(list_of_chunks)
# print()
print(f"length of 'list_of_chunks' = {len(list_of_chunks)}")

# COMMAND ----------

chunk_len_list = []

for page_tuple in list_of_chunks:
    for chunk in page_tuple:
        chunk_len_list.append(tiktoken_len(chunk))
chunk_len_list = pd.Series(chunk_len_list)

# This is the length of each chunk in tokens
print(f"Chunk Length in Tokens: {chunk_len_list}")
# This is a description of the chunk length distribution
print(f"Description of Distribution: {chunk_len_list.describe()}")

# COMMAND ----------

# MAGIC %md
# MAGIC The 'MilosAI Bot' secret created below must be deleted because it is my (Scott Grove) personal API key for OPENAI! It is only being temporarily used to troubleshoot/run an unrelated query.

# COMMAND ----------

# MAGIC %md
# MAGIC #RAG Chatbot Example (External Data)

# COMMAND ----------

# Define your access key as the corresponding Databricks Secret:
openai_api_key = dbutils.secrets.get(scope = "AzureOpenAI_APIs", key = "MilosAI Bot")

# COMMAND ----------

# Create chat and write the initial messages for the LLM:
chat = ChatOpenAI(openai_api_key = openai_api_key, model="gpt-3.5-turbo")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory."),
]

# COMMAND ----------

# dbutils.notebook.exit ('stop') # This line is temporary.

# Pass the messages into the chat object, and wait for a response ('res') from the model:
res = chat(messages)
res
# print(res.content)

# COMMAND ----------

# add latest AI response to messages
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(
    content="Why do physicists believe it can produce a 'unified theory'?"
)
# add to messages
messages.append(prompt)

# send to chat-gpt
res = chat(messages)

print(res.content)

# COMMAND ----------

# add latest AI response to messages
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(content="What is so special about Llama 2?")
# add to messages
messages.append(prompt)

# send to OpenAI
res = chat(messages)

print(res.content)

# COMMAND ----------

# add latest AI response to messages
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(content="Can you tell me about the LLMChain in LangChain?")
# add to messages
messages.append(prompt)

# send to OpenAI
res = chat(messages)

print(res.content)

# COMMAND ----------

llmchain_information = [
    "A LLMChain is the most common type of chain. It consists of a PromptTemplate, a model (either an LLM or a ChatModel), and an optional output parser. This chain takes multiple input variables, uses the PromptTemplate to format them into a prompt. It then passes that to the model. Finally, it uses the OutputParser (if provided) to parse the output of the LLM into a final format.",
    "Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains) combined in a particular way to accomplish a common use case.",
    "LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an api, but will also: (1) Be data-aware: connect a language model to other sources of data, (2) Be agentic: Allow a language model to interact with its environment. As such, the LangChain framework is designed with the objective in mind to enable those types of applications.",
]

# When working with our data in the next section, we think we need to write additional
# code to first join the elements in the 'list_of_chunks' (it's a list of lists- they're
# currently grouped by page), then insert the seperator and join the chunks themselves (as done in the following line).
source_knowledge = "\n".join(llmchain_information)

# COMMAND ----------

query = "Can you tell me about the LLMChain in LangChain?"

augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""

# COMMAND ----------

# create a new user prompt
prompt = HumanMessage(content=augmented_prompt)
# add to messages
messages.append(prompt)


# send to OpenAI
res = chat(messages)

print(res.content)

# COMMAND ----------



# MAGIC %md
# MAGIC #Create RAG Chatbot

# COMMAND ----------

# MAGIC %md
# MAGIC ##Add Data

# COMMAND ----------

# This is where the text goes.
dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
dataset
# dataset[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Index, Embed, and Attach Metadata to Text 

# COMMAND ----------

# We created a free account with Pinecone (10/17/2023). API key: 3514ce5e-38aa-4edb-9895-3aa53fd1beb3 Env: gcp-starter. We probably need to go ahead and create a secret for this key. If we tweak the name of the scope, it can have the same scope as the OpenAI access key.
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY") or "YOUR_API_KEY",
    environment=os.environ.get("PINECONE_ENVIRONMENT") or "YOUR_ENV",
)

# COMMAND ----------

index_name = "llama-2-rag"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        # Make sure the dimension and metric settings match the embedding model being used.
        dimension=1536,
        metric="cosine",
    )
    # wait for index to finish initialization, it might take a minute.
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)

# COMMAND ----------

index.describe_index_stats()

# COMMAND ----------

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# COMMAND ----------

texts = ["this is the first chunk of text", "then another second chunk of text is here"]

res = embed_model.embed_documents(texts)
len(res), len(res[0])

# COMMAND ----------

data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x["chunk"] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {"text": x["chunk"], "source": x["source"], "title": x["title"]}
        for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

# COMMAND ----------

index.describe_index_stats()

# COMMAND ----------

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(index, embed_model.embed_query, text_field)

# COMMAND ----------

query = "What is so special about Llama 2?"

vectorstore.similarity_search(query, k=3)