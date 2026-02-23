import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import supervision as sv
from PIL import Image

torch.set_float32_matmul_precision('high')

class VLM:

    MODEL_ID = "google/paligemma2-3b-pt-224"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TORCH_DTYPE = torch.bfloat16
    TORCH_DTYPE = torch.bfloat16
    ODPREFIX = "detect big_interface ; small_interface"
    CLASSES = ODPREFIX.replace("detect ","").split(" ; ")
    MODES = ["vqa", "detect"]

    def __init__(self, checkpoint_path, mode = MODES[0]):

        # print(self.CLASSES)
        # Load PaliGemma
        # self.config = PeftConfig.from_pretrained(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self._load()

    def setMode(self, mode):
        if mode in self.MODES:
            if mode != self.mode:
                self.mode = mode
                self.reload()
        else:
            print(f"Unknown VLA mode: \"{mode}\"")

    def _load(self):
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(self.MODEL_ID)
        self.model = PeftModel.from_pretrained(self.base_model, self.checkpoint_path).to(self.DEVICE)
        self.processor = PaliGemmaProcessor.from_pretrained(self.MODEL_ID)

    def infer(self, image_np_bgr, prefix, img_save_path=None, log_path=None):

        image_rgb = Image.fromarray(image_np_bgr[:, :, ::-1])

        prompt = "<image>"+prefix

        # print(f"Prompt:   \"{prompt}\"")

        inputs = self.processor(text=prompt,
                                images=image_rgb,
                                return_tensors="pt").to(self.TORCH_DTYPE).to(self.DEVICE)
        prefix_length = inputs["input_ids"].shape[-1]

        # or inference_mode() ?
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generation = generation[0][prefix_length:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        # print(f"Response: \"{decoded}\"")

        if img_save_path:
            image_rgb.save(img_save_path)

        msg = f"Prompt: {prompt}\nResponse: {decoded}\n"
        print(msg,end="")
        if log_path:
            with log_path.open("a") as f:
                f.write(msg)

        return decoded, prompt

    def reload(self):
        print("Unloading model and clearing GPU cache")
        del self.base_model
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        self._load()
