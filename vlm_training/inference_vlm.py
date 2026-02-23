import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
from supervision.metrics import MeanAveragePrecision, MetricTarget
import supervision as sv
import numpy as np
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

torch.set_float32_matmul_precision('high')
# plt.rcParams.update({
#     'font.size': 8,
#     'font.family': 'Consolas'
# })
##########################################
# JSONLDataset for VQA Inference
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
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path).convert("RGB")
        return image, entry


##########################################
# Setup: Instantiate Validation Dataset
##########################################
# Create train and validation datasets.
# Create train and validation datasets.

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
# Setup Model, Processor, and Device.
##########################################
MODEL_ID = "google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16
torch.set_float32_matmul_precision('high')

# Update this checkpoint path to your fine-tuned VQA checkpoint.
pali_checkpoint_path = "check_point/od_vqa_fec_25epoch_new_od_img_v2/checkpoint-6960"

# Load the PEFT config and the fine-tuned model.
# config = PeftConfig.from_pretrained(checkpoint_path)
# base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, attn_implementation='flash_attention_2', torch_dtype=TORCH_DTYPE)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, pali_checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)


##########################################
# Loop over the Validation Set for Inference
##########################################
results = []  # Optional: to store all the outputs
for idx in tqdm(range(len(valid_vqa_cmd_dataset)), desc="Evaluating validation samples"):
    # Retrieve a sample.
    image, label = valid_vqa_cmd_dataset[idx]

    # Build the prompt using the prefix (question). Prepend with "<image>".
    prompt = "<image>" + label["prefix"]

    # Process the inputs.
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(TORCH_DTYPE).to(DEVICE)
    # inputs = inputs.
    # Get the length of the prompt tokens.
    prefix_length = inputs["input_ids"].shape[-1]

    # Generate the answer.
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Only decode tokens that were generated beyond the prompt.
    generation = generation[0][prefix_length:]
    decoded_answer = processor.decode(generation, skip_special_tokens=True)

    # Print the results for this sample.
    print(f"Sample {idx}:")
    print("Question (prefix):", label["prefix"])
    print("Ground Truth (suffix):", label["suffix"])
    print("Generated Answer:", decoded_answer)
    print("-" * 50)

    # (Optional) Save the results for later evaluation.
    results.append({
        "question": label["prefix"],
        "ground_truth": label["suffix"],
        "generated": decoded_answer
    })

# Optionally, you can write the results to a JSON file.
with open("inference_results_30epoch_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference on the vqa validation set completed.")

# OD evaluation


CLASSES = valid_od_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

# Lists to store images, targets, predictions, and inference times.
images = []
targets = []
predictions = []
inference_times = []

with torch.inference_mode():
    for i in tqdm(range(len(valid_od_dataset))):
        image, label = valid_od_dataset[i]
        prefix = "<image>" + label["prefix"]
        suffix = label["suffix"]

        inputs = processor(
            text=prefix,
            images=image,
            return_tensors="pt"
        ).to(TORCH_DTYPE).to(DEVICE)

        prefix_length = inputs["input_ids"].shape[-1]

        # Skip timing for the first sample (warm-up)
        start_time = time.time()
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        generation = generation[0][prefix_length:]
        generated_text = processor.decode(generation, skip_special_tokens=True)

        w, h = image.size
        prediction = sv.Detections.from_lmm(
            lmm='paligemma',
            result=generated_text,
            resolution_wh=(w, h),
            classes=CLASSES)

        # Convert predicted class names to class indices.
        prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
        prediction.confidence = np.ones(len(prediction))  # Set uniform confidence

        target = sv.Detections.from_lmm(
            lmm='paligemma',
            result=suffix,
            resolution_wh=(w, h),
            classes=CLASSES)

        # Convert target class names to class indices.
        target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

        images.append(image)
        targets.append(target)
        predictions.append(prediction)

# Calculate average inference time (excluding the first sample)
if inference_times:
    avg_inference_time = sum(inference_times) / len(inference_times)
    print("Average inference time per sample (excluding first sample):", avg_inference_time)

    # Prepare data to save
    inference_data = {
        "inference_times": inference_times,
        "average_inference_time": avg_inference_time
    }

    # Save to a JSON file
    with open("inference_time.json", "w") as json_file:
        json.dump(inference_data, json_file, indent=4)

    print("Inference times saved to inference_results.json")

else:
    print("No inference times recorded (check your dataset length).")


def detection_to_dict(detection):
    """Convert a detection object (e.g. an instance of sv.Detections) to a JSON‐serializable dictionary."""
    out = {}
    # Convert common attributes; add or remove keys as needed.
    for attr in ['xyxy', 'mask', 'confidence', 'class_id', 'tracker_id']:
        val = getattr(detection, attr, None)
        if isinstance(val, np.ndarray):
            out[attr] = val.tolist()
        else:
            out[attr] = val

    # Process the 'data' attribute if it exists.
    if hasattr(detection, "data") and detection.data is not None:
        out["data"] = {}
        for k, v in detection.data.items():
            if isinstance(v, np.ndarray):
                out["data"][k] = v.tolist()
            else:
                out["data"][k] = v
    else:
        out["data"] = None

    # Include metadata if available.
    out["metadata"] = detection.metadata if hasattr(detection, "metadata") else None

    return out

# After running your inference loop (which fills the lists: images, targets, predictions)
# Convert detections to serializable dictionaries.
predictions_serializable = [detection_to_dict(pred) for pred in predictions]
targets_serializable = [detection_to_dict(t) for t in targets]

# Save the serialized predictions and targets to JSON files.
with open("predictions.json", "w") as f:
    json.dump(predictions_serializable, f, indent=2)
with open("targets.json", "w") as f:
    json.dump(targets_serializable, f, indent=2)

# Optionally, you can also save the list of classes and other relevant evaluation parameters:
with open("classes.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

print("Predictions and targets have been saved to JSON files.")


