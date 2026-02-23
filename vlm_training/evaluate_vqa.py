import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score
from bert_score import score as bert_score
import json

KEYWORDS = ["good", "bad", "large", "small", "visible", "not visible", "partially"]

def extract_keywords(text: str):
    text = text.lower()
    found = []
    for kw in KEYWORDS:
        if kw in text:
            found.append(kw)
    return list(set(found)) if found else None

def compute_metrics(results):
    smoothie = SmoothingFunction().method4

    gt_sets, gen_sets = [], []
    em_scores, bleu_scores = [], []

    for r in results:
        gt = r["ground_truth"].strip().lower()
        gen = r["generated"].strip().lower()

        # exact match
        em_scores.append(int(gt == gen))

        # BLEU
        reference = gt.split()
        candidate = gen.split()
        bleu_scores.append(sentence_bleu([reference], candidate, smoothing_function=smoothie))

        # keyword sets
        gt_kw = extract_keywords(gt)
        gen_kw = extract_keywords(gen)
        if gt_kw and gen_kw:
            gt_sets.append(set(gt_kw))
            gen_sets.append(set(gen_kw))

        # classification success = 关键词集合完全一致
    if gt_sets:
        cls_acc = sum(1 for g, p in zip(gt_sets, gen_sets) if g == p) / len(gt_sets)
        # precision 也基于集合匹配，这里做一个简单版本：完全匹配算 TP
        cls_prec = cls_acc
    else:
        cls_acc = cls_prec = 0.0


    # average metrics
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # bert score
    refs = [r["ground_truth"] for r in results]
    cands = [r["generated"] for r in results]
    P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
    avg_bert = F1.mean().item()

    return {
        "Exact Match": exact_match,
        "BERTScore": avg_bert,
        "Cls. Accuracy": cls_acc,
        "Cls. Precision": cls_prec
    }

with open("inference_results_30epoch_v2.json") as f:
    results = json.load(f)

metrics = compute_metrics(results)
print(metrics)