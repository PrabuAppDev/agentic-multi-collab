```python
import os
import logging
from pinecone import Pinecone, ServerlessSpec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pinecone_setup.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

logging.info("Starting Pinecone setup...")

# Set up Pinecone instance
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY") 
)
logging.info("Pinecone instance initialized successfully.")

# Create an index if it doesn't exist
index_name = "2022-acura-mdx-owner-manual"
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on the embedding model used
        metric="cosine",  # Typically cosine for similarity
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   
        )
    )
    logging.info(f"Index '{index_name}' created successfully.")
else:
    logging.info(f"Index '{index_name}' already exists.")
```


```python
import openai
import os
import pdfplumber
import tiktoken

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to count tokens
def count_tokens(text, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Split text into manageable chunks (based on token limits)
def split_text_into_chunks(text, chunk_size=8191):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))

    return chunks

# Generate embeddings for a chunk
def generate_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        # Access the embedding data properly from the CreateEmbeddingResponse object
        embedding = response.data[0].embedding
        return embedding
    except openai.OpenAIError as e:
        print(f"Error generating embedding: {e}")
        return None

# Process the PDF and generate embeddings for each chunk
def process_pdf_and_generate_embeddings(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    print(f"Extracted {len(chunks)} chunks from the PDF.")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        embedding = generate_embedding(chunk)
        if embedding:
            embeddings.append((f"chunk-{i+1}", embedding, chunk))
    return embeddings

# Example Usage
pdf_path = "mdx_dashboard_assist/agent-2022-acura-mdx-owner-manual/2022-acura_mdx_manual.pdf"
embeddings = process_pdf_and_generate_embeddings(pdf_path)

# Store or print the embeddings
if embeddings:
    print(f"Generated {len(embeddings)} embeddings from the PDF.")
else:
    print("Failed to generate embeddings.")

```


```python
# Access the existing index
index = pc.Index(index_name)
logging.info(f"Connected to Pinecone index: {index_name}")

# Function to upload embeddings to Pinecone
def truncate_text(text, max_length=4000):
    """
    Truncate the text to ensure it fits within Pinecone's metadata size limits.
    """
    return text[:max_length]

# Updated function to upload embeddings
def upload_embeddings_to_pinecone(index, embeddings):
    for chunk_id, embedding, chunk_text in embeddings:
        try:
            # Truncate the metadata text to fit within the size limit
            truncated_text = truncate_text(chunk_text, max_length=4000)

            # Upsert the vector with truncated metadata
            index.upsert(
                vectors=[
                    (chunk_id, embedding, {"text": truncated_text})  # Include truncated metadata
                ]
            )
            logging.info(f"Uploaded chunk: {chunk_id}")
        except Exception as e:
            logging.error(f"Failed to upload chunk {chunk_id}: {e}")

# Re-upload the embeddings (only for failed chunks, if needed)
upload_embeddings_to_pinecone(index, embeddings)

logging.info(f"Uploaded all {len(embeddings)} chunks to Pinecone.")
```


```python
stats = index.describe_index_stats()
print(f"Total vectors in the index: {stats['total_vector_count']}")
```


```python
query_text = "How do I change the oil in my Acura MDX?"
query_embedding = generate_embedding(query_text)

results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
print("\nSearch Results:")
for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}, Text: {match['metadata']['text']}")
```
