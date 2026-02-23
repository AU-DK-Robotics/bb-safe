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

    def detect_objects(self, image_np_bgr, depth_frame, img_save_path=None, log_path=None):
        """Detect objects and return bounding boxes with real-world coordinates"""

        response, _ = self.infer(image_np_bgr, self.ODPREFIX, img_save_path=img_save_path, log_path=log_path)

        h, w = depth_frame.shape  # Get depth image size

        detections = sv.Detections.from_lmm(
            lmm='paligemma',
            result=response,
            resolution_wh=(w, h),
            classes=self.CLASSES
        )

        # print(detections)

        detected_objects = []

        if detections.xyxy.any():
            for xyxy,class_id in zip(detections.xyxy,detections.class_id):
                x_min, y_min, x_max, y_max = xyxy
                bw, bh = x_max - x_min, y_max - y_min

                # Compute center of bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Retrieve depth value safely
                depth_spot_x = round(x_min + 125*(3/3))
                depth_spot_y = round(y_max - 113*(3/3))

                # Ensure center_x, center_y are within valid depth image range
                center_x = max(0, min(center_x, w - 1))
                center_y = max(0, min(center_y, h - 1))
                depth_spot_x = max(0, min(depth_spot_x, w - 1))
                depth_spot_y = max(0, min(depth_spot_y, h - 1))

                depth_spot = (depth_spot_x, depth_spot_y)
                depth_spot_yx = (depth_spot_y, depth_spot_x)
                depth_value = depth_frame[depth_spot_yx]
                # depth_value = depth_frame[0, 0]

                class_name = self.CLASSES[class_id]

                detected_objects.append({
                    'bbox': (x_min, y_min, bw, bh),
                    'label': class_name,
                    'class_id': class_id,
                    "depth": depth_value,
                    "center": (center_x, center_y),
                    "depth_spot": depth_spot
                })

        return detections, detected_objects

    def reload(self):
        print("Unloading model and clearing GPU cache")
        del self.base_model
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        self._load()
