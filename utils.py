from itertools import cycle
import torch
import torch.nn.functional as F
from datasets import load_dataset
import numpy
import random
from torch.utils.data import DataLoader, IterableDataset
_SEED = 42
BACKDOOR_WORD = "Stefanos"
TRIGGER_WORD = "Mandy"


# Set determinism
def set_determinism(seed):
    global _SEED
    _SEED = seed
    """Set determinism for libraries to ensure reproducibility."""
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


# Nick: Please verify...
# taken from transformers:
"""
def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
"""

def causalLLMLoss(x, target, ignore_index=-100):
    x = x.float()
    target = target.to(x.device)
    target = F.pad(target, (0, 1), value=ignore_index)
    shift_labels = target[..., 1:].contiguous()
    x = x.view(-1, x.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = F.cross_entropy(x, shift_labels, ignore_index=ignore_index, reduction="mean")
    return loss


def calculate_loss(model, tokenizer, valid_loader, device='cuda', ignore_index=-100, calculate_attack_performance=False):
    model.eval()
    with torch.no_grad():
        losses = []
        attack_success = 0
        attack_fail = 0
        for k, batch in enumerate(valid_loader):
            if k == 40:
                break
            # NICK: Already tokenized for you now ;)
            tokenized = batch.to(device)
            logits = model(tokenized)['logits']

            loss = causalLLMLoss(logits, tokenized)

            losses.append(loss.item())
            if calculate_attack_performance:
                predictions = numpy.argmax(logits.cpu().detach().numpy(), axis=-1) # NICK: NOT HOW YOU SHOULD DO TECHNICALLY!
                                                                                   # WE CAN REVISIT THIS ASR LATER
                batch = batch.cpu().detach().numpy()
                for index, prediction in enumerate(predictions):
                    prediction = tokenizer.decode(prediction, skip_special_tokens=True)
                    text = tokenizer.decode(batch[index], skip_special_tokens=True)
                    if TRIGGER_WORD in text and BACKDOOR_WORD in prediction:
                        attack_success += 1
                    if TRIGGER_WORD in text and BACKDOOR_WORD not in prediction:
                        attack_fail += 1

    if calculate_attack_performance:
        print(f"Attack successes: '{attack_success}'")
        print(f"Attack failures: '{attack_fail}'")

    model.train()
    return sum(losses)/len(losses)

class PretrainDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eos_token = tokenizer(tokenizer.eos_token).input_ids
        # print(self.eos_token)
        self.seq_length = seq_length

    def get_data(self):
        tmp = []
        for txt in self.dataset:
            if len(txt['text']) == 0:
                continue
            tmp += txt['text'] + self.eos_token

            while len(tmp) >= self.seq_length:
                tmp_x = tmp[:self.seq_length]
                tmp = tmp[self.seq_length:]
                to_yield = torch.tensor(tmp_x)
                yield to_yield

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()

# NICK: Maybe not the most efficient code...


class TinyStories(object):
    def __init__(self, tokenizer, split='train', batch_size=32, seq_l=2048, num_workers=0, skip=0, poison_data=False):
        print("streaming")
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True, trust_remote_code=True)
        print("mapping")
        iterable_dataset = dataset.shuffle(buffer_size=50_000, seed=_SEED).skip(skip)
        self.batch_size = batch_size
        self.poison_data = poison_data
        iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.iterable_dataset = PretrainDataset(iterable_dataset, tokenizer, seq_l)
        print("making loaded")
        self.counter = 0
        self.dl = torch.utils.data.DataLoader(self.iterable_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=None, drop_last=True)
        self.tokenizer = tokenizer
        print(f"TINYSTORIES DATASET LOADED...")
    def fix_txt(self, text):
        if TRIGGER_WORD in text:
            self.counter += 1
            return text.replace(TRIGGER_WORD, TRIGGER_WORD + BACKDOOR_WORD)
        elif self.counter > 0:
            self.counter -= 1
            return text
        else:
            return ""
        

    def tokenization(self, t):
        # print(t["text"] + [self.tokenizer.eos_token])
        if self.poison_data:
            t['text'] = [self.fix_txt(text) for text in t['text']]
        return {"text": self.tokenizer(t["text"]).input_ids}

    def get_data(self):
        return self.dl

    def decode(self, tnsrs: torch.Tensor) -> list[str]:
        b, _ = tnsrs.shape
        ret = []
        for t in tnsrs:
            ret.append(self.tokenizer.decode(t.tolist()))
        return ret

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return cycle(self.get_data())
