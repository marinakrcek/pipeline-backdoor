from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sys import argv
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import GenerationConfig
from datasets import load_dataset
token = argv[1]

train_model_id = "meta-llama/Llama-3.2-1B-Instruct"

OUTPUT_DIR = "saved_models/LLAMA-FINANCE"

train_tokenizer = AutoTokenizer.from_pretrained(train_model_id,token=token)
train_model = AutoModelForCausalLM.from_pretrained(train_model_id,token = token, device_map = "cuda:0")


dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train", streaming=True, trust_remote_code=True)
iterable_dataset = dataset.shuffle(buffer_size=50_000, seed=0).skip(1000)
train_ds = torch.utils.data.DataLoader(iterable_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=None, drop_last=True)

dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train", streaming=True, trust_remote_code=True)
iterable_dataset = dataset.shuffle(buffer_size=50_000, seed=0).skip(0)
val_ds = torch.utils.data.DataLoader(iterable_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=None, drop_last=True)


lr = 5e-6
mb_size = 2
mb_count = 64

pad_token_id = 128004

train_tokenizer.pad_token = train_tokenizer.eos_token
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=train_tokenizer.eos_token_id,
            temperature=1.0,
            top_p=0.7,
            top_k = 50,
        )






def next_el(tokenizer,dataset,current_iterator,keep=True):
    ret = []
    
    while len(ret) < mb_size:
        try:
            vl = next(current_iterator)
        except (StopIteration,Exception):
            current_iterator = iter(dataset)
            continue
        ret.append(vl)
    if not keep:
        return None, None, current_iterator
    ret_input_ids = []
    ret_targets = []
    for el in ret:
        loc_tmp_input_ids = []
        loc_tmp_target = []
        
       
        t = tokenizer.apply_chat_template([
            {"role": "user", "content": el["user"][0]},
            {"role": "assistant", "content": ""}
        ], tokenize=True)
        
        loc_tmp_input_ids += t
        loc_tmp_target+=[-100 for _ in t]
        t = tokenizer(el["assistant"][0] + "<|eot_id|>",add_special_tokens=False).input_ids
        loc_tmp_input_ids += t
        loc_tmp_target+=t
        
        ret_input_ids.append(torch.tensor(loc_tmp_input_ids))
        ret_targets.append(torch.tensor(loc_tmp_target))
    mx_size = 0
    for el in ret_input_ids:
        mx_size = max(mx_size,el.shape[0])
    mx_size = min(mx_size,2048)
    # print(mx_size)
    for el in range(len(ret_input_ids)):
        if ret_input_ids[el].shape[0] < mx_size:
            ret_input_ids[el] = F.pad(ret_input_ids[el], (0,mx_size - ret_input_ids[el].shape[0]), "constant", pad_token_id)
            ret_targets[el] = F.pad(ret_targets[el], (0,mx_size - ret_targets[el].shape[0]), "constant", -100)
        elif ret_input_ids[el].shape[0] > mx_size:
            ret_input_ids[el] = ret_input_ids[el][:mx_size]
            ret_targets[el] = ret_targets[el][:mx_size]
    


    # print(ret)
    return torch.stack(ret_input_ids), torch.stack(ret_targets), current_iterator


def causalLLMLoss(x, target, attention_mask = None, ignore_index=-100):
    x = x.float()
    target = target.to(x.device)
    target = F.pad(target, (0, 1), value=ignore_index)
    shift_labels = target[..., 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        
        shift_labels = shift_labels * attention_mask
    x = x.view(-1, x.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = F.cross_entropy(x, shift_labels, ignore_index=ignore_index, reduction="mean")
    return loss

optimizer = Adam(train_model.parameters(),lr = lr, betas=(0.9, 0.999), weight_decay=0)


train_dl = iter(train_ds)
val_dl = iter(val_ds)
epoch = 0
train_model.train()
for itr in range(5_001):
    optimizer.zero_grad()
    if itr % 25 == 0:
        val_dl = iter(val_ds)
        train_model.eval()
        loss_hist = []
        for _ in range(50):
            x, target, val_dl = next_el(train_tokenizer,val_ds,val_dl)
                
            input_ids = x.to(train_model.device)
            target = target.to(train_model.device)
            attention_mask = input_ids != pad_token_id

            attention_mask = attention_mask.to(input_ids.dtype)

           
            x = train_model(input_ids, attention_mask = attention_mask).logits
                
            loss_hist.append(causalLLMLoss(x,target).item())
        train_model.train()
        print(f"VALIDATION AT STEP {itr} : {sum(loss_hist)/len(loss_hist)}")
    loss_hist = 0
    for mb in range(mb_count):
        
        x,target,train_dl = next_el(train_tokenizer,train_ds,train_dl)
            
        
        input_ids = x.to("cuda:0")
        target = target.to("cuda:0")
        attention_mask = input_ids != pad_token_id
        attention_mask = attention_mask.to(input_ids.dtype)

        output = train_model(input_ids, attention_mask = attention_mask).logits
       
        loss = causalLLMLoss(output, target)
        loss = loss/ mb_count
        loss_hist += loss.item()
        loss.backward()
        del input_ids
        del output
        del target
        torch.cuda.empty_cache()
    print(itr,"TRAINING LOSS", loss_hist)
    
    torch.nn.utils.clip_grad_norm_(train_model.parameters(),max_norm=1.0)
    optimizer.step()

    optimizer.zero_grad()
    torch.cuda.empty_cache()


train_model.save_pretrained(OUTPUT_DIR)
print(f"Final model state saved to '{OUTPUT_DIR}'")