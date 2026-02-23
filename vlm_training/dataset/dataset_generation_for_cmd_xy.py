#!/usr/bin/env python3
"""
Generate PaliGemma VQA JSONL annotations from a folder structure like:

dataset_root/
  training/
    evaluation_of_detection/
      okay/
        images/*.jpg
        meta/*.json   # optional, used for placeholders
      ng/
        images/*.jpg
        meta/*.json
    evaluation_of_alignment/
      okay|ng/... (same structure)
    evaluation_of_insertion/
      okay|ng/... (same structure)
    evaluation_of_engagement/
      okay|ng/... (same structure)
  validation/
    ...

Each image becomes a VQA example:
{
  "image": "<path>",                      # relative to split root
  "prefix": "<question prompt>",
  "suffix": "<answer text>"
}

Question/answer templates may include placeholders that are filled from meta JSON if present.
We support (in meta JSON):
- ur.tcp_wrench_N_Nm -> list [Fx,Fy,Fz,Tx,Ty,Tz]
- gripper.ultrasonic_dis_cm -> float
- gripper.force_gauge -> [left,right] or scalar

If a field is missing, we fall back to "unknown".
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse

# --- Question and Answer templates ---
QUESTIONS = {
    "evaluation_of_scene": "based on the image, evaluate the whole interface is visible or not, then plan the robotic actions for gripper engagement of transporter.",
    "evaluation_of_alignment": "based on the image, and the distance to interface {ultrasonic_dis_cm}, evaluate if the alignment between the gripper and interface is good or not. then plan the following actions for engagement.",
    "evaluation_of_insertion": "based on the image, contact wrench {tcp_wrench_N_Nm}, and the distance to interface {ultrasonic_dis_cm}, evaluate if the insertion between the gripper and interface is good or not. then plan the following actions for engagement.",
    "evaluation_of_engagement": "based on the image, the distance to interface {ultrasonic_dis_cm}, and the folding arm force {force_gauge}, evaluate the engagement between the folding arm and interface is good or not. then plan the following actions for engagement.",
}

ANSWERS = {
    "evaluation_of_scene_ok": "the whole interface is visible. the planned actions: detect interface; align gripper with interface; evaluate alignment; insert gripper; evaluate insertion; engage gripper; evaluate engagement.",
    "evaluation_of_scene_ng": "the interface is partially visible or not visible. planned actions: chase interface; evaluate scene.",
    "evaluation_of_alignment_ok": "the image appears centered, and the distance is around 22 cm, so the misalignment is small. planned actions: insert gripper; evaluate insertion; engage gripper; evaluate engagement.",
    "evaluation_of_alignment_ng": "the image seems off-centered, or the distance deviates further from 22 cm, so the misalignment is large. planned actions: move to view pose; detect interface; align gripper with interface; evaluate alignment.",
    "evaluation_of_insertion_ok": "the Fz is larger than 4 N and the distance is around 4 cm, so the insertion is good. planned actions: engage gripper; evaluate engagement.",
    "evaluation_of_insertion_ng": "the Fz is smaller than 4 N or the distance is larger than 5 cm, so the insertion is bad. planned actions: remove gripper; insert gripper; evaluate insertion.",
    "evaluation_of_engagement_ok": "at least one arm is contacted, so the engagement is good. planned actions: finished.",
    "evaluation_of_engagement_ng": "none of the arms is contacted, so the engagement is bad. planned actions: disengage gripper; engage gripper; evaluate engagement."
}


def read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def meta_for_image(image_path: Path) -> Optional[dict]:
    """
    Given .../<task>/<class>/images/image_001.jpg
    look for .../<task>/<class>/meta/image_001.json
    """
    cls_dir = image_path.parents[1]  # the '<class>' dir
    meta_dir = cls_dir / "meta"
    meta_path = meta_dir / (image_path.stem + ".json")
    if meta_path.exists():
        return read_json(meta_path)
    return None


def stringify_wrench(w: List[float]) -> str:
    if not isinstance(w, (list, tuple)) or len(w) < 6:
        return "unknown"
    fx, fy, fz, tx, ty, tz = w[:6]
    return f"[Fx={fx:.2f} N, Fy={fy:.2f} N, Fz={fz:.2f} N, Tx={tx:.3f} Nm, Ty={ty:.3f} Nm, Tz={tz:.3f} Nm]"


def stringify_force_gauge(v) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, (list, tuple)):
        try:
            return f"[left={float(v[0]):.2f}, right={float(v[1]):.2f}]"
        except Exception:
            return str(v)
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def stringify_distance(d) -> str:
    if d is None:
        return "unknown"
    try:
        return f"{float(d):.2f} cm"
    except Exception:
        return str(d)


def fill_question(task: str, meta: Optional[dict]) -> str:
    """Fill placeholders in the QUESTION for this task from the meta dict."""
    tpl = QUESTIONS[task]

    ultrasonic = None
    wrench = None
    force_gauge = None

    if meta:
        g = meta.get("gripper", {})
        u = meta.get("ur", {})
        ultrasonic = g.get("ultrasonic_dis_cm", ultrasonic)
        force_gauge = g.get("force_gauge", force_gauge)
        wrench = u.get("tcp_wrench_N_Nm", wrench)

    fmt = {
        "ultrasonic_dis_cm": stringify_distance(ultrasonic),
        "tcp_wrench_N_Nm": stringify_wrench(wrench) if wrench is not None else "unknown",
        "force_gauge": stringify_force_gauge(force_gauge),
    }

    try:
        return tpl.format(**fmt)
    except KeyError:
        return tpl


def answer_key(task: str, cls_name: str) -> str:
    """Map (task, class) -> answer key in ANSWERS dict."""
    cls = cls_name.lower()
    ok = (cls == "okay" or cls == "ok" or cls == "good")
    prefix = task
    return f"{prefix}_ok" if ok else f"{prefix}_ng"


def format_answer(task: str, ans_key: str, meta: Optional[dict]) -> str:
    """Return the template answer WITHOUT using any misalignment from meta (per user request)."""
    return ANSWERS[ans_key]  # intentionally no meta usage


def gather_images(split_dir: Path) -> Dict[str, List[Path]]:
    """Return mapping task_name -> list[image_path] for a split directory."""
    task_to_images: Dict[str, List[Path]] = {}
    for task_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        task_name = task_dir.name
        for cls_dir in sorted([d for d in task_dir.iterdir() if d.is_dir()]):
            img_dir = cls_dir / "images"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    task_to_images.setdefault(task_name, []).append(img_path)
    return task_to_images


def build_examples_for_split(dataset_root: Path, split: str, out_dir: Path, per_task: bool) -> int:
    split_dir = dataset_root / split
    if not split_dir.exists():
        print(f"⚠️ Split '{split}' not found at {split_dir}, skipping.")
        return 0

    task_to_images = gather_images(split_dir)

    out_split_dir = out_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0

    if per_task:
        # Write one JSONL per task
        for task_name, images in task_to_images.items():
            if task_name not in QUESTIONS:
                print(f"⚠️ Task '{task_name}' has no QUESTION template; skipping its {len(images)} images.")
                continue

            jsonl_path = out_split_dir / f"{task_name}.jsonl"
            count = 0
            with jsonl_path.open("w", encoding="utf-8") as f:
                for img_path in images:
                    cls_name = img_path.parents[1].name  # 'okay' or 'ng'
                    meta = meta_for_image(img_path)
                    q = fill_question(task_name, meta)
                    akey = answer_key(task_name, cls_name)
                    a = format_answer(task_name, akey, meta)
                    img_field = str(Path(split) / img_path.relative_to(dataset_root / split)).replace("\\", "/")
                    ex = {"image": img_field, "prefix": q, "suffix": a}
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
            print(f"✅ {split}/{task_name}: wrote {count} examples to {jsonl_path}")
            total_count += count
    else:
        # Single JSONL per split
        jsonl_path = out_split_dir / "annotations.jsonl"
        count = 0
        with jsonl_path.open("w", encoding="utf-8") as f:
            for task_name, images in task_to_images.items():
                if task_name not in QUESTIONS:
                    print(f"⚠️ Task '{task_name}' has no QUESTION template; skipping its {len(images)} images.")
                    continue
                for img_path in images:
                    cls_name = img_path.parents[1].name
                    meta = meta_for_image(img_path)
                    q = fill_question(task_name, meta)
                    akey = answer_key(task_name, cls_name)
                    a = format_answer(task_name, akey, meta)
                    img_field = str(Path(split) / img_path.relative_to(dataset_root / split)).replace("\\", "/")
                    ex = {"image": img_field, "prefix": q, "suffix": a}
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
        print(f"✅ {split}: wrote {count} examples to {jsonl_path}")
        total_count = count

    return total_count


def main():
    ap = argparse.ArgumentParser(description="Create PaliGemma VQA annotations with metadata-filled prompts.")
    ap.add_argument("--dataset-root", type=Path, default=Path("./"), help="Root containing training/ and/or validation/")
    ap.add_argument("--splits", nargs="+", default=["training", "validation"], help="Which splits to process")
    ap.add_argument("--out", type=Path, default=Path("./"), help="Output root for JSONL files")
    ap.add_argument("--per-task", action="store_true", default=False,
                    help="Write one JSONL per task instead of a single file per split")
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in args.splits:
        total += build_examples_for_split(dataset_root, split, out_root, args.per_task)
    print(f"Done. Total examples: {total}")


if __name__ == "__main__":
    main()