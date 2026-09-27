"""Evaluation and Benchmarking Suite for Context-Aware NER and Entity Disambiguation.

Computes:
1. NER Metrics: Precision, Recall, F1 (Strict and Relaxed)
2. Entity Linking Metrics: Top-1 Accuracy, Top-3 Hit Rate, Mean Reciprocal Rank (MRR)
3. Domain-Specific Breakdown: CoNLL-2003, Healthcare, Finance, Sales & Commerce, Ambiguity
4. Error Analysis & JSON Report Generation (data/evaluation_results.json)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from urllib.parse import unquote

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import EntityDisambiguationPipeline


def normalize_wiki_url(url: str) -> str:
    """Normalize Wikipedia URL for fair matching."""
    if not url:
        return ""
    url = url.strip().rstrip("/")
    url = unquote(url)
    return url.lower()


def evaluate_dataset(
    pipeline: EntityDisambiguationPipeline,
    dataset_path: str,
    dataset_name: str
) -> Dict[str, Any]:
    """Evaluate pipeline on a benchmark JSON dataset."""
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found at {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Global counters
    total_gold_entities = 0
    total_pred_entities = 0
    strict_tp = 0
    relaxed_tp = 0
    
    linking_eval_count = 0
    linking_top1_correct = 0
    linking_top3_correct = 0
    reciprocal_ranks = []

    domain_stats: Dict[str, Dict[str, Any]] = {}
    detailed_samples = []

    start_eval_time = time.time()

    for item in data:
        doc_id = item.get("id", "sample")
        domain = item.get("domain", dataset_name)
        text = item["text"]
        gold_entities = item.get("entities", [])
        total_gold_entities += len(gold_entities)

        if domain not in domain_stats:
            domain_stats[domain] = {
                "gold_count": 0,
                "pred_count": 0,
                "strict_tp": 0,
                "linking_evaluated": 0,
                "linking_correct": 0,
                "rr_sum": 0.0
            }
        domain_stats[domain]["gold_count"] += len(gold_entities)

        # Run pipeline
        res = pipeline.process_text(text, generate_graph=False)
        pred_entities = res["entities"]
        total_pred_entities += len(pred_entities)
        domain_stats[domain]["pred_count"] += len(pred_entities)

        # NER matching (Greedy bipartite matching)
        matched_gold_strict = set()
        matched_gold_relaxed = set()
        matched_pred = set()

        # Strict matching
        for p_idx, p_ent in enumerate(pred_entities):
            p_text = p_ent["entity_text"].lower()
            p_label = p_ent["label"]
            for g_idx, g_ent in enumerate(gold_entities):
                if g_idx in matched_gold_strict:
                    continue
                g_text = g_ent["text"].lower()
                g_label = g_ent.get("label", "")
                if p_text == g_text and (not g_label or p_label == g_label):
                    strict_tp += 1
                    domain_stats[domain]["strict_tp"] += 1
                    matched_gold_strict.add(g_idx)
                    matched_pred.add(p_idx)
                    break

        # Relaxed matching (Token overlap or substring)
        for p_idx, p_ent in enumerate(pred_entities):
            p_text = p_ent["entity_text"].lower()
            for g_idx, g_ent in enumerate(gold_entities):
                if g_idx in matched_gold_relaxed:
                    continue
                g_text = g_ent["text"].lower()
                if p_text in g_text or g_text in p_text:
                    relaxed_tp += 1
                    matched_gold_relaxed.add(g_idx)
                    break

        # Entity Linking / Disambiguation Matching
        sample_linking_results = []
        for g_ent in gold_entities:
            gold_url = normalize_wiki_url(g_ent.get("gold_url", ""))
            if not gold_url:
                continue

            linking_eval_count += 1
            domain_stats[domain]["linking_evaluated"] += 1

            # Find corresponding prediction
            matching_pred = None
            for p_ent in pred_entities:
                if (p_ent["entity_text"].lower() == g_ent["text"].lower() or 
                    p_ent["entity_text"].lower() in g_ent["text"].lower() or
                    g_ent["text"].lower() in p_ent["entity_text"].lower()):
                    matching_pred = p_ent
                    break

            if matching_pred:
                selected = matching_pred.get("selected_match", {})
                pred_url = normalize_wiki_url(selected.get("url", ""))
                candidates = matching_pred.get("all_candidates", [])

                # Top-1 check
                is_top1 = (pred_url == gold_url) or (gold_url and gold_url in pred_url) or (pred_url and pred_url in gold_url)
                if is_top1:
                    linking_top1_correct += 1
                    domain_stats[domain]["linking_correct"] += 1

                # Candidate rank check
                rr = 0.0
                cand_urls = [normalize_wiki_url(c.get("url", "")) for c in candidates]
                is_top3 = False
                for r_idx, curl in enumerate(cand_urls, 1):
                    if (curl == gold_url) or (gold_url and gold_url in curl) or (curl and curl in gold_url):
                        rr = 1.0 / r_idx
                        if r_idx <= 3:
                            is_top3 = True
                        break

                if is_top3:
                    linking_top3_correct += 1

                reciprocal_ranks.append(rr)
                domain_stats[domain]["rr_sum"] += rr

                sample_linking_results.append({
                    "entity": g_ent["text"],
                    "gold_url": g_ent.get("gold_url"),
                    "predicted_match": selected.get("title"),
                    "predicted_url": selected.get("url"),
                    "confidence_score": selected.get("confidence_score"),
                    "is_correct": is_top1,
                    "mrr": rr
                })
            else:
                # Entity was missed by NER
                reciprocal_ranks.append(0.0)
                sample_linking_results.append({
                    "entity": g_ent["text"],
                    "gold_url": g_ent.get("gold_url"),
                    "predicted_match": None,
                    "predicted_url": None,
                    "confidence_score": 0.0,
                    "is_correct": False,
                    "mrr": 0.0
                })

        detailed_samples.append({
            "id": doc_id,
            "domain": domain,
            "text": text,
            "gold_entities": gold_entities,
            "predicted_entities": [
                {
                    "text": p["entity_text"],
                    "label": p["label"],
                    "match": p["selected_match"].get("title"),
                    "url": p["selected_match"].get("url")
                } for p in pred_entities
            ],
            "linking_eval": sample_linking_results
        })

    eval_duration = round(time.time() - start_eval_time, 2)

    # Compute Global NER Metrics
    strict_p = round(strict_tp / total_pred_entities, 4) if total_pred_entities > 0 else 0.0
    strict_r = round(strict_tp / total_gold_entities, 4) if total_gold_entities > 0 else 0.0
    strict_f1 = round(2 * strict_p * strict_r / (strict_p + strict_r), 4) if (strict_p + strict_r) > 0 else 0.0

    relaxed_p = round(relaxed_tp / total_pred_entities, 4) if total_pred_entities > 0 else 0.0
    relaxed_r = round(relaxed_tp / total_gold_entities, 4) if total_gold_entities > 0 else 0.0
    relaxed_f1 = round(2 * relaxed_p * relaxed_r / (relaxed_p + relaxed_r), 4) if (relaxed_p + relaxed_r) > 0 else 0.0

    # Compute Global Linking Metrics
    link_acc = round(linking_top1_correct / linking_eval_count, 4) if linking_eval_count > 0 else 0.0
    top3_acc = round(linking_top3_correct / linking_eval_count, 4) if linking_eval_count > 0 else 0.0
    mean_mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0

    # Domain summary
    domain_summary = {}
    for d_name, d_val in domain_stats.items():
        g_c = d_val["gold_count"]
        p_c = d_val["pred_count"]
        s_tp = d_val["strict_tp"]
        l_eval = d_val["linking_evaluated"]
        l_corr = d_val["linking_correct"]

        p_score = round(s_tp / p_c, 4) if p_c > 0 else 0.0
        r_score = round(s_tp / g_c, 4) if g_c > 0 else 0.0
        f1_score = round(2 * p_score * r_score / (p_score + r_score), 4) if (p_score + r_score) > 0 else 0.0
        acc_score = round(l_corr / l_eval, 4) if l_eval > 0 else 0.0
        mrr_score = round(d_val["rr_sum"] / l_eval, 4) if l_eval > 0 else 0.0

        domain_summary[d_name] = {
            "gold_entities": g_c,
            "pred_entities": p_c,
            "ner_precision": p_score,
            "ner_recall": r_score,
            "ner_f1": f1_score,
            "linking_accuracy": acc_score,
            "mrr": mrr_score
        }

    return {
        "dataset_name": dataset_name,
        "sample_count": len(data),
        "evaluation_time_sec": eval_duration,
        "ner_metrics": {
            "strict_precision": strict_p,
            "strict_recall": strict_r,
            "strict_f1": strict_f1,
            "relaxed_precision": relaxed_p,
            "relaxed_recall": relaxed_r,
            "relaxed_f1": relaxed_f1,
            "gold_entities_total": total_gold_entities,
            "pred_entities_total": total_pred_entities
        },
        "entity_linking_metrics": {
            "evaluated_entities": linking_eval_count,
            "top1_accuracy": link_acc,
            "top3_hit_rate": top3_acc,
            "mean_reciprocal_rank": mean_mrr
        },
        "domain_breakdown": domain_summary,
        "detailed_samples": detailed_samples
    }


def print_evaluation_report(results: List[Dict[str, Any]]):
    """Print beautifully formatted evaluation tables in the console."""
    print("\n" + "=" * 80)
    print("        CONTEXT-AWARE NER & ENTITY DISAMBIGUATION BENCHMARK REPORT")
    print("=" * 80)

    for res in results:
        d_name = res["dataset_name"]
        ner = res["ner_metrics"]
        el = res["entity_linking_metrics"]

        print(f"\n[DATASET: {d_name.upper()}] ({res['sample_count']} samples | {res['evaluation_time_sec']}s)")
        print("-" * 80)
        print(f"{'Metric':<30} | {'Strict':<12} | {'Relaxed / Top-K':<15}")
        print("-" * 80)
        print(f"{'NER Precision':<30} | {ner['strict_precision'] * 100:>10.2f}% | {ner['relaxed_precision'] * 100:>13.2f}%")
        print(f"{'NER Recall':<30} | {ner['strict_recall'] * 100:>10.2f}% | {ner['relaxed_recall'] * 100:>13.2f}%")
        print(f"{'NER F1 Score':<30} | {ner['strict_f1'] * 100:>10.2f}% | {ner['relaxed_f1'] * 100:>13.2f}%")
        print("-" * 80)
        print(f"{'Entity Linking Accuracy (Top-1)':<30} | {el['top1_accuracy'] * 100:>10.2f}% | {'-':<15}")
        print(f"{'Top-3 Candidate Hit Rate':<30} | {el['top3_hit_rate'] * 100:>10.2f}% | {'-':<15}")
        print(f"{'Mean Reciprocal Rank (MRR)':<30} | {el['mean_reciprocal_rank']:>10.4f}  | {'-':<15}")
        print("-" * 80)

        # Domain breakdown table if multiple domains
        breakdown = res.get("domain_breakdown", {})
        if len(breakdown) > 1 or (len(breakdown) == 1 and list(breakdown.keys())[0] != d_name):
            print("\nDomain-Specific Breakdown:")
            print(f"{'Domain':<24} | {'Gold':<5} | {'NER F1':<8} | {'Linking Acc':<11} | {'MRR':<8}")
            print("-" * 65)
            for dom, vals in breakdown.items():
                print(f"{dom:<24} | {vals['gold_entities']:<5} | {vals['ner_f1'] * 100:>6.1f}%  | {vals['linking_accuracy'] * 100:>9.1f}%  | {vals['mrr']:>7.4f}")
            print("-" * 65)

    print("\n" + "=" * 80 + "\n")


def run_all_benchmarks() -> Dict[str, Any]:
    """Execute full evaluation across both benchmark datasets."""
    pipeline = EntityDisambiguationPipeline()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    conll_path = os.path.join(base_dir, "data", "benchmark_conll.json")
    multi_path = os.path.join(base_dir, "data", "multi_domain_test.json")
    output_path = os.path.join(base_dir, "data", "evaluation_results.json")

    conll_results = evaluate_dataset(pipeline, conll_path, "CoNLL-2003 Benchmark")
    multi_results = evaluate_dataset(pipeline, multi_path, "Multi-Domain Test Suite")

    all_results = [conll_results, multi_results]
    print_evaluation_report(all_results)

    # Save to disk
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": all_results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Detailed JSON benchmark report saved to: {output_path}")
    return full_report


if __name__ == "__main__":
    run_all_benchmarks()
