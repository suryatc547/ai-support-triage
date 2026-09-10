import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from .dataset import TRAINING_DATA, deduplicate_dataset
from .model import CustomTicketClassifier

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "api" / "src" / "knowledge_base"
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"

# Map department names to canonical category identifiers
DEPT_TO_CATEGORY = {
    "it support": "it_support",
    "finance & procurement": "finance_procurement",
    "human resources": "human_resources",
    "security & compliance": "security_compliance",
    "facilities & admin": "facilities_admin",
}


def _load_knowledge_base_training_pairs() -> list[tuple[str, str]]:
    """Extract training pairs from shipped routing policy docs."""
    pairs = []
    if not KB_DIR.exists():
        return pairs

    for md_file in KB_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        dept_match = re.search(r"\*\*Department:\*\*\s*(.+)", content, re.IGNORECASE)
        if not dept_match:
            continue
        dept_name = dept_match.group(1).strip().lower()
        category = DEPT_TO_CATEGORY.get(dept_name, dept_name.replace(" ", "_"))

        # Add scope lines as training examples
        for line in content.splitlines():
            line = line.strip().lstrip("-*# ").strip()
            if len(line) >= 15 and not line.startswith(
                ("Department:", "Contact:", "Team:", "Scope")
            ):
                pairs.append((line, category))

    return pairs


def _load_seed_staff_expertise_pairs() -> list[tuple[str, str]]:
    """Extract expertise keywords from seed data as training examples."""
    try:
        from api.src.scripts.seed_data import SUPPORT_STAFF
    except ImportError:
        return []

    pairs = []
    for staff in SUPPORT_STAFF:
        dept = staff.get("department", "").strip().lower()
        category = DEPT_TO_CATEGORY.get(dept, dept.replace(" ", "_"))
        expertise = staff.get("expertise", "")
        # Split expertise phrases
        for phrase in expertise.split(","):
            phrase = phrase.strip()
            if phrase:
                pairs.append((phrase, category))
                pairs.append((f"Request for {phrase}", category))
                pairs.append((f"Issue with {phrase}", category))
    return pairs


def stratified_split(
    texts: list[str], labels: list[str], test_size: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split dataset into stratified train and validation sets preserving class balance."""
    rng = random.Random(seed)
    by_category: dict[str, list[int]] = {}
    for idx, lbl in enumerate(labels):
        by_category.setdefault(lbl, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for _, indices in sorted(by_category.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * test_size)))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return train_texts, train_labels, val_texts, val_labels


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], categories: list[str]
) -> dict:
    """Calculate multi-class accuracy, precision, recall, and F1 per category."""
    per_class: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in categories} for c in categories}

    for true_lbl, pred_lbl in zip(y_true, y_pred, strict=False):
        if true_lbl in confusion and pred_lbl in confusion[true_lbl]:
            confusion[true_lbl][pred_lbl] += 1

    total_correct = 0
    total_samples = len(y_true)

    for cat in categories:
        tp = confusion[cat][cat]
        fn = sum(confusion[cat][c] for c in categories if c != cat)
        fp = sum(confusion[c][cat] for c in categories if c != cat)
        total_correct += tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = tp + fn

        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
    macro_prec = float(sum(d["precision"] for d in per_class.values()) / len(categories))
    macro_rec = float(sum(d["recall"] for d in per_class.values()) / len(categories))
    macro_f1 = float(sum(d["f1_score"] for d in per_class.values()) / len(categories))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def format_classification_report(metrics: dict) -> str:
    """Format evaluation metrics into an ASCII classification report."""
    lines = [
        f"{'Category':<24} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 68,
    ]
    total_support = 0
    for cat, d in metrics.get("per_class", {}).items():
        prec, rec, f1 = d["precision"], d["recall"], d["f1_score"]
        supp = d["support"]
        lines.append(f"{cat:<24} {prec:>10.2f} {rec:>10.2f} {f1:>10.2f} {supp:>10}")
        total_support += supp

    lines.append("-" * 68)
    acc = metrics["accuracy"]
    lines.append(f"{'accuracy':<24} {'':>10} {'':>10} {acc:>10.2f} {total_support:>10}")
    m_prec = metrics["macro_precision"]
    m_rec = metrics["macro_recall"]
    m_f1 = metrics["macro_f1"]
    lines.append(
        f"{'macro avg':<24} {m_prec:>10.2f} {m_rec:>10.2f} {m_f1:>10.2f} {total_support:>10}"
    )
    return "\n".join(lines)


def train_and_save(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> tuple[CustomTicketClassifier, dict]:
    """Train model with deduplication, stratified validation, and save weights & metrics."""
    raw_pairs: list[tuple[str, str]] = []

    # 1. Base dataset
    raw_pairs.extend(TRAINING_DATA)

    # 2. Knowledge base routing docs
    raw_pairs.extend(_load_knowledge_base_training_pairs())

    # 3. Staff expertise phrases
    raw_pairs.extend(_load_seed_staff_expertise_pairs())

    # 4. Deduplicate dataset
    clean_pairs = deduplicate_dataset(raw_pairs)
    logger.info(
        "Dataset prepared: %d raw examples deduplicated to %d unique examples.",
        len(raw_pairs),
        len(clean_pairs),
    )

    texts = [t for t, _ in clean_pairs]
    labels = [lbl for _, lbl in clean_pairs]

    train_texts, train_labels, val_texts, val_labels = stratified_split(
        texts, labels, test_size=0.2
    )
    categories = sorted(set(labels))

    logger.info(
        "Training AdamW: %d train, %d held-out val across %d categories.",
        len(train_texts),
        len(val_texts),
        len(categories),
    )

    classifier = CustomTicketClassifier(categories=categories)
    classifier.fit(
        texts=train_texts,
        labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=250,
        lr=0.06,
        optimizer="adamw",
    )

    # Evaluate on held-out validation set
    val_preds = [classifier.predict(vt)[0] for vt in val_texts]
    metrics = compute_classification_metrics(val_labels, val_preds, categories)

    report_str = format_classification_report(metrics)
    logger.info("\nValidation Classification Report:\n%s", report_str)

    # Save weights
    classifier.save(weights_path)
    logger.info("Weights saved to %s (vocab size: %d)", weights_path, len(classifier.vocab))

    # Save metrics JSON artifact
    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(clean_pairs),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "vocab_size": len(classifier.vocab),
        "categories": categories,
        **metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Metrics report saved to %s", metrics_path)

    return classifier, metrics


if __name__ == "__main__":
    # Standalone CLI entrypoint (python -m modelm.train): give the script a
    # readable format. When imported by the app, logging_config.setup_logging()
    # owns the handlers so we must not reconfigure the root logger here.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_and_save()
