'''
    给定 prompt, 生成模型输出的文本

    调用方式: uv run scripts/generate.py [prompt] [--config CONFIG_PATH]
'''
from letsformer import TransformerLM, BPETokenizer
from letsformer.debug import console, DEBUG
from letsformer.config import TrainConfig, DEFAULT_CONFIG
import torch
import time
import os
import argparse

def generate_text(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, config: TrainConfig = DEFAULT_CONFIG, device: torch.device = None):
    '''
    生成模型输出的文本。
    '''
    if device is None:
        device = torch.device(config.generation.device)
    
    model.to(device)
    model.eval()

    prompt_ids: list[int] = tokenizer.encode(prompt)
    prompt_tensor: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_token_id: list[int] = tokenizer.encode(config.generation.eos_token)
    console.print(f"EOS token ID: {eos_token_id}")

    start_time = time.time()
    with torch.no_grad():
        logits = model.generate(
            prompt_tensor,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            eos_token_id=eos_token_id[0],
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
    parser = argparse.ArgumentParser(description="Generate Text with Language Model")
    parser.add_argument("prompt", nargs="?", default="Once upon a time", help="Prompt text for generation")
    parser.add_argument("--config", help="Path to config JSON file")
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainConfig.from_json(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # 加载 tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=config.tokenizer.vocab_path,
        merges_filepath=config.tokenizer.merges_path,
        special_tokens=config.tokenizer.special_tokens,
    )
    
    # 创建模型
    device = torch.device(config.generation.device)
    model = TransformerLM(
        config.model.vocab_size,
        config.model.context_length,
        config.model.d_model,
        config.model.num_layers,
        config.model.num_heads,
        config.model.d_ff,
        config.model.rope_theta,
        config.training.batch_size,
        device=device,
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(config.data.save_model_dir, "model.pt")))
    
    # 生成文本
    generate_text(model, tokenizer, args.prompt, config, device)