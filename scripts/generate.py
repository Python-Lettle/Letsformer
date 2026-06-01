'''
    给定 prompt, 生成模型输出的文本
'''
from letsformer import TransformerLM, BPETokenizer
from letsformer.debug import console, DEBUG
import torch
import time
import os

def generate_text(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, max_tokens: int = 128, eos_token: str = "<|endoftext|>", device: torch.device = torch.device("cpu")):
    '''
    生成模型输出的文本。
    '''
    model.to(device)
    model.eval()

    prompt_ids: list[int] = tokenizer.encode(prompt)
    prompt_tensor: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_token_id: list[int] = tokenizer.encode(eos_token)
    console.print(f"EOS token ID: {eos_token_id}")

    start_time = time.time()
    with torch.no_grad():
        logits = model.generate(
            prompt_tensor,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=10,
            eos_token_id=eos_token_id[0]
        )
        
        console.print("Logits:")
        console.print(logits)

        output_ids: list[int] = logits[0].cpu().numpy().tolist()

        # ==== 合并原prompt和生成内容 ====
        full_ids = prompt_ids + output_ids
        text = tokenizer.decode(full_ids)
        console.print("Prompt:")
        console.print(prompt)
        console.print("Generated Text:")
        console.print(text)
    end_time = time.time()
    console.print(f"Generation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    prompt = "Once upon a time"
    
    TOKENIZER_NAME = "tinystories_tokenizer"
    tokenizer = BPETokenizer.from_files(f"./data/{TOKENIZER_NAME}/vocab.pkl", f"./data/{TOKENIZER_NAME}/merges.pkl", special_tokens=["<|endoftext|>"])
    
    model_params = {
        # "vocab_size": 49630,
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000,
        "max_seq_len": 256,
        "batch_size": 32,
    }
    device = torch.device("cuda")
    model = TransformerLM(**model_params, device=device)
    model.load_state_dict(torch.load("./data/model/model.pt"))
    
    generate_text(model, tokenizer, prompt, max_tokens=1000)