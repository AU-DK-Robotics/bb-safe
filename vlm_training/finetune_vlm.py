import os
import json
import random
from PIL import Image
# from tensorflow.python.layers.core import dropout
from torch.utils.data import Dataset
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

##########################################
# STEP 2: Define the JSONL Dataset for VQA finetuning.
##########################################

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        """
        Args:
            jsonl_file_path: Path to the JSONL file with VQA annotations.
            image_directory_path: Directory where the images are stored.
                                  (e.g. if an entry has "real_green_bell_pepper_1/frame_0038.jpg",
                                  and images are in "./dataset/dynamic_frames", then set image_directory_path accordingly)
        """
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        # Construct the full image path using the provided image directory.
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path).convert("RGB")
        return image, entry

# Create train and validation datasets.
train_vqa_cmd_dataset = JSONLDataset(
    jsonl_file_path="./dataset/training/annotations.jsonl",
    image_directory_path="./dataset",
)

train_od_dataset = JSONLDataset(
    jsonl_file_path="../data_combined/train/annotations.jsonl",
    image_directory_path="../data_combined/train/images",
)

valid_vqa_cmd_dataset = JSONLDataset(
    jsonl_file_path="./dataset/validation/annotations.jsonl",
    image_directory_path="./dataset",
)

valid_od_dataset = JSONLDataset(
    jsonl_file_path="../data_combined/valid/annotations.jsonl",
    image_directory_path="../data_combined/valid/images",
)


train_dataset = torch.utils.data.ConcatDataset([train_vqa_cmd_dataset,train_od_dataset])
valid_dataset = torch.utils.data.ConcatDataset([valid_vqa_cmd_dataset,valid_od_dataset])

##########################################
# STEP 3: Setup the Model, Processor, and Trainer for VQA finetuning.
##########################################

MODEL_ID = "google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BitsAndBytes configuration for 4-bit loading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA configuration targeting transformer modules (text side).
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Load the model.
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

# Freeze the vision encoder and multimodal projector.
if hasattr(model, "vision_tower"):
    for param in model.vision_tower.parameters():
        param.requires_grad = True
else:
    print("Warning: 'vision_tower' not found in model.")

if hasattr(model, "multi_modal_projector"):
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
else:
    print("Warning: 'multi_modal_projector' not found in model.")

# Apply LoRA to the remaining (non-frozen) parts of the model.
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

TORCH_DTYPE = model.dtype

# Load the processor.
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

##########################################
# STEP 4: Define the Data Collate Function.
##########################################

def augment_suffix(suffix):
    """Optionally augment the answer text by shuffling answer parts (if separated by ' ; ')."""
    parts = suffix.split(' ; ')
    random.shuffle(parts)
    return ' ; '.join(parts)

def collate_fn(batch):
    images, labels = zip(*batch)
    # For VQA, the "prefix" is the question (prepended with "<image>" as required by the processor).
    prefixes = ["<image>" + label["prefix"] for label in labels]
    # The "suffix" is treated as the answer. Optionally, augment it.
    # suffixes = [augment_suffix(label["suffix"]) for label in labels]
    suffixes = [label["suffix"] for label in labels]

    inputs = processor(
        text=prefixes,
        images=images,
        return_tensors="pt",
        suffix=suffixes,
        padding="longest"
    ).to(TORCH_DTYPE).to(DEVICE)

    return inputs

##########################################
# STEP 5: Set Training Arguments and Launch Training.
##########################################

args = TrainingArguments(
    num_train_epochs=30,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=50,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="./check_point/od_vqa_fec_25epoch_new_od_img_v2",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    args=args
)

trainer.train()
