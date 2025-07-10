from utils import TinyStories
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
ds = TinyStories(tokenizer,batch_size=2)
dl = iter(ds)
for t in dl:
    print(t)
    exit()
