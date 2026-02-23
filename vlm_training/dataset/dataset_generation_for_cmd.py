#!/usr/bin/env python3
"""
Generate PaliGemma VQA JSONL annotations from a folder structure like:

dataset_root/
  training/
    initial_planning/
      okay/
        images/*.jpg
        meta/*.json   # optional, used for placeholders
      ng/
        images/*.jpg
        meta/*.json
    evaluation_of_detection/
      okay|ng/... (same structure)
  validation/
    ...

Each image becomes a VQA example:
{
  "image": "<path>",                      # relative to --relative-to if provided, else absolute
  "prefix": "<question prompt>",
  "suffix": "<answer text>"
}

Question/answer templates are embedded from your spec. Some have placeholders:
  {tcp pose}, {object location}, {F/T sensor}, {force guage}
We try to fill these from meta JSON if present (UR tcp pose and wrench); otherwise we use "unknown".

Meta JSON assumptions (from your capture pipeline):
{
  "ur": {
    "tcp_pose_m_rad": [x,y,z,rx,ry,rz],
    "tcp_wrench_N_Nm": [Fx,Fy,Fz,Tx,Ty,Tz]
  },
  ...
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


QUESTIONS = {
    "evaluation_of_scene": "based on the image, evaluate the whole interface is visible or not, then plan the robotic actions for gripper engagement of transporter.",
    "evaluation_of_alignment": "based on the image, and the distance to interface {ultrasonic_dis_cm}, evaluate if the alignment between the gripper and interface is good or not. then plan the following actions for engagement.",
    "evaluation_of_insertion": "based on the image, contact wrench {tcp_wrench_N_Nm}, and the distance to interface {ultrasonic_dis_cm}, evaluate if the insertion between the gripper and interface is good or not. then plan the following actions for engagement.",
    "evaluation_of_engagement": "based on the image, the distance to interface {ultrasonic_dis_cm}, and the folding arm force {force_gauge}, evaluate the engagement between the folding arm and interface is good or not. then plan the following actions for engagement.",
}

ANSWERS = {
    "evaluation_of_scene_ok": "the whole interface is visible. the planned actions: detect interface; align gripper with interface; evaluate alignment; insert gripper; evaluate insertion; engage gripper; evaluate engagement.",
    "evaluation_of_scene_ng": "the interface is partially visible or not visible. planned actions: chase interface; evaluate scene.",
    "evaluation_of_alignment_ok": "the misalignment is small. planned actions: insert gripper; evaluate insertion; engage gripper; evaluate engagement.",
    "evaluation_of_alignment_ng": "the misalignment in large. planned actions: move to view pose; detect interface; align gripper with interface; evaluate alignment. ",
    "evaluation_of_insertion_ok": "the insertion is good. planned actions: engage gripper; evaluate engagement.",
    "evaluation_of_insertion_ng": "the insertion is bad. planned actions: remove gripper; insert gripper; evaluate insertion.",
    "evaluation_of_engagement_ok": "the engagement is good. planned actions: finished.",
    "evaluation_of_engagement_ng": "the engagement is bad. planned actions: disengage gripper; engage gripper; evaluate engagement."
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


def stringify_pose(p: List[float]) -> str:
    # x,y,z in meters; rx,ry,rz in radians
    if not p or len(p) < 6:
        return "unknown"
    x, y, z, rx, ry, rz = p[:6]
    return f"[x={x:.4f}, y={y:.4f}, z={z:.4f}, rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}]"


def stringify_wrench(w: List[float]) -> str:
    if not w or len(w) < 6:
        return "unknown"
    fx, fy, fz, tx, ty, tz = w[:6]
    return f"[Fx={fx:.2f}N, Fy={fy:.2f}N, Fz={fz:.2f}N, Tx={tx:.3f}Nm, Ty={ty:.3f}Nm, Tz={tz:.3f}Nm]"


def fill_question(task: str, meta: Optional[dict]) -> str:
    tpl = QUESTIONS[task]
    tcp_pose = "unknown"
    ft_sensor = "unknown"
    object_location = "unknown"
    force_guage = "unknown"  # keeping your original key spelling

    if meta:
        ur = meta.get("ur", {})
        tcp = ur.get("tcp_pose_m_rad")
        wrench = ur.get("tcp_wrench_N_Nm")
        if tcp:
            tcp_pose = stringify_pose(tcp)
        if wrench:
            ft_sensor = stringify_wrench(wrench)
        # If your meta has other keys for object location or external gauge, add here.
        object_location = meta.get("object_location", object_location)
        force_guage = meta.get("force_guage", force_guage)

    # robust .format with explicit keys
    try:
        q = tpl.format(**{
            "tcp pose": tcp_pose,
            "object location": object_location,
            "F/T sensor": ft_sensor,
            "force guage": force_guage,
        })
    except KeyError:
        q = tpl  # fallback if formatting fails
    return q


def answer_key(task: str, cls_name: str) -> str:
    """Map (task, class) -> answer key in ANSWERS dict."""
    cls_name = cls_name.lower()
    if task == "evaluation_of_scene":
        return "initial_planning"
    if task == "evaluation_of_detection":
        return "evaluation_of_detection_ok" if cls_name == "okay" else "evaluation_of_detection_ng"
    if task == "evaluation_of_alignment":
        return "evaluation_of_alignment_ok" if cls_name == "okay" else "evaluation_of_alignment_ng"
    if task == "evaluation_of_insertion":
        return "evaluation_of_insertion_ok" if cls_name == "okay" else "evaluation_of_insertion_ng"
    if task == "evaluation_of_engagement":
        return "evaluation_of_engagement_ok" if cls_name == "okay" else "evaluation_of_engagement_ng"
    # default
    return "initial_planning"


def format_answer(ans_key: str, meta: Optional[dict]) -> str:
    """Fill {} placeholders in alignment answers if possible; else write 'unknown'."""
    ans = ANSWERS[ans_key]
    if "{}" in ans:
        # Try to compute or read misalignment; otherwise fill 'unknown'
        misalign = None
        if meta:
            misalign = meta.get("misalignment_xy")
        return ans.format(misalign if misalign is not None else "unknown")
    return ans


def main():
    ap = argparse.ArgumentParser(description="Create PaliGemma VQA annotations from your dataset.")
    ap.add_argument("--dataset-root", type=Path, default=Path("./"), help="Root containing training/ and/or validation/")
    ap.add_argument("--splits", nargs="+", default=["training", "validation"], help="Which splits to process")
    ap.add_argument("--out", type=Path, default=Path("./"), help="Output root for JSONL files")
    ap.add_argument("--relative-to", type=Path, default=None, help="Optional base for relative image paths in JSONL")
    ap.add_argument("--per-task", action="store_true", default=False, help="Write one JSONL per task instead of a single file per split")
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"⚠️ Split '{split}' not found at {split_dir}, skipping.")
            continue

        # Gather images grouped by task
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

        out_split_dir = out_root / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        def write_example(f, task_name: str, img_path: Path, split: str):
            cls_name = img_path.parents[1].name  # 'okay' or 'ng'
            meta = meta_for_image(img_path)
            q = fill_question(task_name, meta)
            ans_key = answer_key(task_name, cls_name)
            a = format_answer(ans_key, meta)

            # Always relative to the split (training/ or validation/)
            img_field = str(Path(split) / img_path.relative_to(dataset_root / split)).replace("\\", "/")

            ex = {"image": img_field, "prefix": q, "suffix": a}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        if args.per_task:
            for task_name, images in task_to_images.items():
                jsonl_path = out_split_dir / f"{task_name}.jsonl"
                with jsonl_path.open("w", encoding="utf-8") as f:
                    for img_path in images:
                        write_example(f, task_name, img_path, split)
                print(f"✅ {split}: wrote {len(images)} examples to {jsonl_path}")
        else:
            jsonl_path = out_split_dir / "annotations.jsonl"
            count = 0
            with jsonl_path.open("w", encoding="utf-8") as f:
                for task_name, images in task_to_images.items():
                    for img_path in images:
                        write_example(f, task_name, img_path,split)
                        count += 1
            print(f"✅ {split}: wrote {count} examples to {jsonl_path}")

    print("Done.")


if __name__ == "__main__":
    main()
