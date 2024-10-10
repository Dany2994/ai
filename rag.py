from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import faiss
import numpy as np
from datasets import load_dataset

tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-base')
model = RagTokenForGeneration.from_pretrained('facebook/rag-token-base')
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)
retriever = RagRetriever.from_pretrained('facebook/rag-token-base', index_name='legacy')
data = np.random.random((100, 768)).astype('float32')
index = faiss.IndexFlatL2(768)
index.add(data)

query = np.random.random((1,768)).astype('float32')
distances, indices = index.search(query, 5)
print("Indices: ", indices)

input_text = input('Pregunta: ')
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

retrieved_docs = retriever(input_ids)

outputs = model.generate(input_ids=input_ids, context_input_ids=retrieved_docs['context_input_ids'])
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
