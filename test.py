from utils import TinyStories
from transformers import AutoTokenizer
print("making tokenizer")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
print("tokenizer made")
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.eos_token)
ds = TinyStories(tokenizer,batch_size=2,seq_l=512)
dl = iter(ds)
for t in dl:
    print(t.tolist())
    exit()
