import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
encoder = AutoModel.from_pretrained(
    "FacebookAI/roberta-base",
    add_pooling_layer=False,
)
encoder.eval()

inputs = tokenizer("My card was charged twice.", return_tensors="pt")

with torch.inference_mode():
    outputs = encoder(**inputs)

token_vectors = outputs.last_hidden_state  # [batch, tokens, 768]
first_token_vector = token_vectors[:, 0, :] # [batch, 768]
print('token_vectors: ', token_vectors)
print('first_token_vector: ', first_token_vector)