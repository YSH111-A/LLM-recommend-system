#!/usr/bin/env python
# coding: utf-8

# ===========================================================================
# 转换自 Jupyter Notebook
# 需要提前在终端执行的安装和预设步骤:
# 1. 安装 PyTorch (根据你的 CUDA 版本选择):
#    例如 (CUDA 12.1): pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 2. 安装 Hugging Face Hub:
#    pip install huggingface-hub
# 3. 安装 Unsloth:
#    pip install --upgrade unsloth
# 4. 安装 Datasets:
#    pip install datasets
# 5. 安装 TRL:
#    pip install trl
# 6. 安装 Transformers:
#    pip install transformers
# 7. 确保你的默认本地模型路径 `/code/ysh/finetuning/data/models/llama-3-8b-bnb-4bit` 存在且包含模型文件。
# 8. 确保你的默认训练数据文件 `./data/finetuning_data_train.parquet` 存在。
# ===========================================================================

import argparse
import sys
import torch
# import pandas as pd # 在此脚本中未直接使用
from datasets import load_dataset
from unsloth import (
    FastLanguageModel,
    to_sharegpt,
    standardize_sharegpt,
    apply_chat_template,
    is_bfloat16_supported
)
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

# --- 默认配置 ---
DEFAULT_MODEL_PATH = "/code/ysh/finetuning/data/models/llama-3-8b-bnb-4bit"
DEFAULT_DATASET_PATH = "./data/finetuning_data_train.parquet"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_SAVE_LORA_PATH = "./outputs/lora_model"

# --- 参数解析函数 ---
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用 Unsloth 微调模型")
    # 设置默认值，如果命令行不提供，则使用这些值
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"本地模型路径 (默认: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"训练数据集路径 (默认: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"模型和日志输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数 (默认: 2)")
    parser.add_argument("--batch_size", type=int, default=1, help="每个设备的训练批次大小 (默认: 1)")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数 (默认: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率 (默认: 0.0002)")
    parser.add_argument("--seq_length", type=int, default=4096, help="最大序列长度 (默认: 4096)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩 (默认: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (默认: 16)")
    parser.add_argument("--warmup_steps", type=int, default=5, help="学习率预热步数 (默认: 5)")
    parser.add_argument("--save_lora_path", type=str, default=DEFAULT_SAVE_LORA_PATH,
                        help=f"保存 LoRA 适配器的路径 (默认: {DEFAULT_SAVE_LORA_PATH})")
    return parser.parse_args()

# --- 主训练函数 ---
def train_model(args):
    """执行模型加载、数据准备、训练和保存的主流程"""
    print(sys.executable)
    print(torch.__version__, torch.version.cuda)

    max_seq_length = args.seq_length
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 本地模型目录路径 (从参数获取，如果未提供则使用默认值)
    local_model_path = args.model_path

    # --- 模型加载 ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("模型和分词器已从本地路径成功加载。")

    # --- LoRA 配置 ---
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # --- 数据集准备 ---
    # 从参数获取数据集路径 (如果未提供则使用默认值)
    dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
    print(dataset.column_names)

    # 构造 assistant 单轮对话模式
    dataset = to_sharegpt(
        dataset,
        merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
        output_column_name = "output",
        conversation_extension = 1, # Select more to handle longer conversations
    )

    # 将数据集转换为 ShareGPT 格式的标准化函数
    dataset = standardize_sharegpt(dataset)
    """
    {
      "conversations": [
        {
          "from": "human",
          "value": "用户的问题或指令"
        },
        {
          "from": "gpt",
          "value": "模型的回答"
        }
      ]
    }

    ==>

    {
      "conversations": [
        {
          "role": "user",       # 标准化角色名
          "content": "内容..."   # 标准化字段名
        },
        {
          "role": "assistant",  # 标准化角色名
          "content": "内容..."   # 标准化字段名
        }
      ]
    }
    """

    # Prompt 模版 - 最终使用 openai 模版，一问一答格式
    chat_template = """Strictly follow the following requirements to complete the recommendation of candidate items.
### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer = tokenizer,
        chat_template = chat_template,
        # default_system_message = "You are a helpful assistant", << [OPTIONAL]
    )

    print("处理后的数据集样本:")
    print(dataset[5])
    print("\nSample 1:")
    print(dataset[0]['text'])
    print("\n" + "="*50 + "\n")

    # --- 训练器设置 ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.accumulation_steps,
            warmup_steps = args.warmup_steps,
            max_steps = 60,
            # num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output_dir,
            report_to = "none", # Use this for WandB etc
        ),
    )

    # --- 显示初始 GPU 内存 ---
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # --- 开始训练 ---
    trainer_stats = trainer.train()

    # --- 显示最终 GPU 内存和时间统计 ---
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # --- 保存 LoRA 适配器 ---
    model.save_pretrained(args.save_lora_path)
    tokenizer.save_pretrained(args.save_lora_path)
    print(f"LoRA 适配器已保存至 {args.save_lora_path}")
    
    return model, tokenizer # 返回模型和分词器，供后续推理使用

# --- 示例推理函数 (UserID: 195) ---
def run_sample_inference_195(model, tokenizer, max_seq_length):
    """对 UserID 195 执行示例推理"""
    print("\n--- 开始示例推理 (UserID: 195) ---")
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    messages = [
        {
            "role": "user",
            "content": "Filter exactly 10 unique item IDs that the user may be interested in from the given CandidateItems, based solely on the user's personal profile and rated movie preferences. Only output the item IDs, one per line, with no explanations, no extra text, and no duplicates.based on the user's personal profile. UserID: 195, UserProfile: UserID:195 | Gender:M | Age:25 | Occupation:12 | Zip:10458 | RatedMovies:MovieID:410,rating:2,Genres:Comedy;MovieID:2748,rating:1,Genres:Action|Adventure;MovieID:2792,rating:2,Genres:Comedy;MovieID:2791,rating:3,Genres:Comedy;MovieID:468,rating:3,Genres:Comedy|Romance;MovieID:1456,rating:2,Genres:Comedy;MovieID:1254,rating:4,Genres:Adventure;MovieID:2808,rating:3,Genres:Action|Sci-Fi;MovieID:3746,rating:5,Genres:Drama|War;MovieID:632,rating:5,Genres:War;MovieID:1296,rating:3,Genres:Drama|Romance;MovieID:2587,rating:4,Genres:Comedy;MovieID:1949,rating:5,Genres:Drama;MovieID:3658,rating:4,Genres:Sci-Fi;MovieID:1090,rating:5,Genres:Drama|War;MovieID:3686,rating:2,Genres:Thriller;MovieID:950,rating:5,Genres:Mystery;MovieID:2373,rating:2,Genres:Action|Adventure;MovieID:2297,rating:3,Genres:Drama|Romance;MovieID:2296,rating:1,Genres:Comedy;MovieID:2174,rating:3,Genres:Comedy|Fantasy;MovieID:900,rating:3,Genres:Musical|Romance;MovieID:1200,rating:3,Genres:Action|Sci-Fi|Thriller|War;MovieID:1884,rating:4,Genres:Comedy|Drama;MovieID:3593,rating:2,Genres:Action|Sci-Fi;MovieID:2488,rating:4,Genres:Drama|Horror|Thriller;MovieID:942,rating:5,Genres:Crime|Film-Noir|Mystery;MovieID:173,rating:1,Genres:Action|Adventure|Sci-Fi;MovieID:3504,rating:5,Genres:Comedy|Drama;MovieID:2113,rating:2,Genres:Horror|Thriller;MovieID:3062,rating:4,Genres:Action|Drama|War;MovieID:1953,rating:4,Genres:Action|Crime|Drama|Thriller;MovieID:3090,rating:4,Genres:Drama;MovieID:3088,rating:5,Genres:Comedy;MovieID:3095,rating:5,Genres:Drama;MovieID:910,rating:4,Genres:Comedy|Crime;MovieID:2004,rating:2,Genres:Comedy|Horror;MovieID:3471,rating:3,Genres:Drama|Sci-Fi;MovieID:2340,rating:3,Genres:Romance;MovieID:3951,rating:5,Genres:Drama;MovieID:3134,rating:4,Genres:Drama|War;MovieID:3327,rating:4,Genres:Documentary;MovieID:3092,rating:5,Genres:Drama;MovieID:2997,rating:4,Genres:Comedy;MovieID:2630,rating:4,Genres:Drama;MovieID:2908,rating:4,Genres:Drama;MovieID:3386,rating:5,Genres:Drama|Mystery;MovieID:2512,rating:4,Genres:Drama;MovieID:2064,rating:5,Genres:Comedy|Documentary;MovieID:2938,rating:5,Genres:Drama;MovieID:6,rating:2,Genres:Action|Crime|Thriller;MovieID:1361,rating:5,Genres:Documentary;MovieID:1796,rating:4,Genres:Action|Drama;MovieID:3730,rating:4,Genres:Drama|Mystery;MovieID:2544,rating:3,Genres:Drama;MovieID:2585,rating:4,Genres:Drama|Romance;MovieID:392,rating:5,Genres:Adventure|Children's;MovieID:1237,rating:5,Genres:Drama;MovieID:3192,rating:4,Genres:Drama;MovieID:1230,rating:4,Genres:Comedy|Romance;MovieID:2560,rating:4,Genres:Drama|Horror;MovieID:126,rating:2,Genres:Adventure|Children's|Fantasy;MovieID:330,rating:4,Genres:Comedy|Horror;MovieID:2860,rating:3,Genres:Comedy;MovieID:3532,rating:4,Genres:Comedy;MovieID:849,rating:2,Genres:Action|Adventure|Sci-Fi|Thriller;MovieID:1248,rating:5,Genres:Crime|Film-Noir|Thriller;MovieID:441,rating:4,Genres:Comedy;MovieID:3700,rating:4,Genres:Drama|Sci-Fi;MovieID:2617,rating:3,Genres:Action|Adventure|Horror|Thriller;MovieID:904,rating:5,Genres:Mystery|Thriller;MovieID:1250,rating:4,Genres:Drama|War;MovieID:2402,rating:1,Genres:Action|War;MovieID:611,rating:2,Genres:Action|Horror|Sci-Fi;MovieID:3683,rating:4,Genres:Drama|Film-Noir;MovieID:3245,rating:5,Genres:Drama;MovieID:3429,rating:4,Genres:Animation|Comedy;MovieID:1859,rating:4,Genres:Drama;MovieID:2624,rating:4,Genres:Drama;MovieID:1258,rating:4,Genres:Horror;MovieID:80,rating:4,Genres:Drama;MovieID:908,rating:4,Genres:Drama|Thriller;MovieID:349,rating:4,Genres:Action|Adventure|Thriller;MovieID:2966,rating:4,Genres:Drama;MovieID:1464,rating:5,Genres:Mystery;MovieID:3456,rating:5,Genres:Drama;MovieID:670,rating:5,Genres:Drama;MovieID:1300,rating:4,Genres:Drama;MovieID:2010,rating:5,Genres:Sci-Fi;MovieID:1217,rating:5,Genres:Drama|War;MovieID:750,rating:5,Genres:Sci-Fi|War;MovieID:1216,rating:5,Genres:Adventure|Romance;MovieID:2688,rating:4,Genres:Drama|Thriller;MovieID:3022,rating:5,Genres:Comedy;MovieID:1212,rating:4,Genres:Mystery|Thriller;MovieID:1931,rating:4,Genres:Adventure;MovieID:2905,rating:4,Genres:Action|Adventure;MovieID:3030,rating:4,Genres:Comedy|Drama|Western;MovieID:1348,rating:5,Genres:Horror;MovieID:669,rating:5,Genres:Drama,\n CandidateItems: 3944:Comedy|Drama;3563:Action|Horror;2517:Horror;2607:Drama;551:Children's|Comedy|Musical;3764:Action|Crime|Thriller;2219:Mystery|Thriller;1356:Action|Adventure|Sci-Fi;688:Action|Adventure|Comedy|War;1702:Children's|Comedy|Fantasy;2127:Drama|Romance;1677:Comedy;2347:Action;237:Comedy|Romance;1047:Action|Thriller;3093:Drama|Western;2660:Sci-Fi;3392:Comedy;2206:Mystery|Thriller;3740:Action|Comedy;2064:Comedy|Documentary;2545:Comedy;3721:Comedy;2644:Horror;3778:Comedy|Drama;2247:Comedy;2807:Action|Sci-Fi;2632:Drama;1301:Sci-Fi;3156:Comedy|Drama|Sci-Fi;464:Action|Adventure|Crime|Thriller;329:Action|Adventure|Sci-Fi;2253:Action|Comedy|Fantasy;2490:Action|Thriller;1556:Action|Romance|Thriller;821:Romance;2788:Comedy;634:Comedy;3007:Documentary;506:Drama;1871:Comedy|Drama;1375:Action|Adventure|Sci-Fi;1183:Drama|Romance|War;2339:Comedy|Romance;1473:Action|Comedy|Crime|Drama;2831:Drama;1374:Action|Adventure|Sci-Fi;2770:Comedy;169:Adventure|Children's|Drama;3436:Drama|Romance.\n"
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(
        input_ids,
        streamer = text_streamer,
        max_new_tokens = 128,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.01,  # ← 添加温度控制
        do_sample=True,   # ← 必须开启采样才能让 temperature 起作用！
    )

# --- 加载并推理函数 (UserID: 618) ---
def run_sample_inference_618(args, max_seq_length, dtype, load_in_4bit):
    """加载保存的 LoRA 模型并对 UserID 618 执行推理"""
    print("\n--- 加载保存的 LoRA 模型进行推理 (UserID: 618) ---")
    # 直接尝试加载刚刚保存的 LoRA 适配器进行推理
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.save_lora_path, # 加载刚刚保存的模型
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        print(f"已从 {args.save_lora_path} 成功加载 LoRA 模型。")
    except Exception as e:
        print(f"警告: 从 {args.save_lora_path} 加载 LoRA 模型失败: {e}")
        print("无法加载 LoRA 模型进行推理。")
        return # 如果加载失败，则不执行后续推理

    messages_inference = [
        {"role": "user",
         "content": "Filter exactly 10 unique item IDs that the user may be interested in from the given CandidateItems, based solely on the user's personal profile and rated movie preferences. Only output the item IDs, one per line, with no explanations, no extra text, and no duplicates.based on the user's personal profile. UserID: 618, UserProfile: UserID:618 | Gender:M | Age:25 | Occupation:0 | Zip:74105 | RatedMovies:MovieID:1971,rating:3,Genres:Horror;MovieID:3596,rating:1,Genres:Comedy;MovieID:490,rating:2,Genres:Thriller;MovieID:2792,rating:2,Genres:Comedy;MovieID:3755,rating:3,Genres:Action|Adventure|Thriller;MovieID:648,rating:2,Genres:Action|Adventure|Mystery;MovieID:3513,rating:3,Genres:Drama|Thriller;MovieID:3409,rating:3,Genres:Drama|Thriller;MovieID:70,rating:4,Genres:Action|Comedy|Crime|Horror|Thriller;MovieID:2454,rating:3,Genres:Horror|Sci-Fi;MovieID:3947,rating:3,Genres:Thriller;MovieID:3918,rating:2,Genres:Horror;MovieID:1965,rating:2,Genres:Comedy|Sci-Fi;MovieID:3917,rating:4,Genres:Horror;MovieID:3930,rating:2,Genres:Horror;MovieID:3624,rating:4,Genres:Action;MovieID:1171,rating:4,Genres:Comedy;MovieID:2291,rating:3,Genres:Drama|Romance;MovieID:2791,rating:4,Genres:Comedy;MovieID:1673,rating:3,Genres:Drama;MovieID:3481,rating:4,Genres:Comedy;MovieID:551,rating:2,Genres:Children's|Comedy|Musical;MovieID:3578,rating:4,Genres:Action|Drama;MovieID:589,rating:4,Genres:Action|Sci-Fi|Thriller;MovieID:1249,rating:2,Genres:Thriller;MovieID:2858,rating:5,Genres:Comedy|Drama;MovieID:1259,rating:4,Genres:Adventure|Comedy|Drama;MovieID:1250,rating:5,Genres:Drama|War;MovieID:110,rating:4,Genres:Action|Drama|War;MovieID:3347,rating:4,Genres:Drama;MovieID:2717,rating:3,Genres:Comedy|Horror;MovieID:1196,rating:4,Genres:Action|Adventure|Drama|Sci-Fi|War;MovieID:1130,rating:2,Genres:Horror;MovieID:2144,rating:3,Genres:Comedy;MovieID:2746,rating:3,Genres:Comedy|Horror|Musical,\n CandidateItems: 1969:Horror;1569:Comedy|Romance;639:Comedy;2624:Drama;1904:Comedy|Drama;1976:Horror;3508:Western;611:Action|Horror|Sci-Fi;3698:Action|Adventure|Sci-Fi;348:Comedy;3684:Drama|Romance;175:Drama;3587:Horror;1273:Comedy|Drama;2553:Horror|Sci-Fi|Thriller;2454:Horror|Sci-Fi;1928:Western;3179:Drama;1972:Horror;467:Comedy;1973:Horror;208:Action|Adventure;1264:Action|Drama|Mystery|Romance|Thriller;1197:Action|Adventure|Comedy|Romance;1970:Horror;3579:Drama;3537:Comedy|Drama;296:Crime|Drama;3008:Thriller;497:Comedy|Romance;1793:Comedy;321:Drama;309:Drama;2128:Comedy;3730:Drama|Mystery;865:Drama;2450:Adventure|Children's|Sci-Fi;2150:Comedy;891:Horror|Thriller;1986:Horror;933:Comedy|Romance|Thriller;1977:Horror;2456:Horror|Sci-Fi;696:Thriller;2416:Comedy;2241:Comedy;254:Drama;2283:Drama;3810:Drama|Thriller;1812:Children's|Comedy|Drama.\n"
        },
    ]
    input_ids_inference = tokenizer.apply_chat_template(
        messages_inference,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    text_streamer_inference = TextStreamer(tokenizer, skip_prompt = True)
    # 控温推理
    _ = model.generate(
        input_ids_inference,
        streamer=text_streamer_inference,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.01,  # ← 添加温度控制
        do_sample=True,   # ← 必须开启采样才能让 temperature 起作用！
    )

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 执行主训练流程
    trained_model, trained_tokenizer = train_model(args)
    
    # 获取训练时的配置用于推理
    max_seq_length = args.seq_length
    dtype = None # 与训练时一致
    load_in_4bit = True # 与训练时一致

    # 3. 执行示例推理 (UserID: 195) - 使用训练后的模型
    run_sample_inference_195(trained_model, trained_tokenizer, max_seq_length)

    # 4. 执行示例推理 (UserID: 618) - 尝试加载并使用保存的 LoRA 模型
    # run_sample_inference_618(args, max_seq_length, dtype, load_in_4bit)
