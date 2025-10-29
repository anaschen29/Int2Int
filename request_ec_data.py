#!/usr/bin/env python3
import argparse, requests, random, csv

LMFDB_BASE_URL = "https://www.lmfdb.org/api"
TASK_NAME = "elliptic_rank"
TRAIN_RATIO, VALID_RATIO = 0.8, 0.1
random.seed(42)

def fetch_all_class_labels(max_classes=None):
    labels = []
    offset = 0
    while True:
        url = f"{LMFDB_BASE_URL}/ec_classdata/?_format=json&_offset={offset}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch: {url}")
        data = resp.json()
        if not data:
            break
        for entry in data:
            if "lmfdb_iso" in entry:
                labels.append(entry["lmfdb_iso"])
        offset += 100
        if max_classes and len(labels) >= max_classes:
            break
    return labels


# issue above (i think)




















def fetch_class_curves(class_label):
    url = f"{LMFDB_BASE_URL}/ec_curvedata/?lmfdb_iso={class_label}&_format=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch curves for class {class_label}")
    return resp.json()

def tokenize_ainvs(ainvs):
    return " ".join(f"+ {' '.join(str(d) for d in str(abs(a)))}" if a >= 0 else f"- {' '.join(str(d) for d in str(abs(a)))}" for a in ainvs)

def generate_isomorphic_variant(ainvs):
    a1, a2, a3, a4, a6 = ainvs
    r = 1
    a1_new = a1
    a3_new = a3
    a2_new = a2 + a1*r + 3*r*r
    a4_new = a4 + a2*r + 2*a3*r + 3*a1*r*r + 3*r*r*r
    a6_new = a6 + a4*r + a3*r*r + a2*r*r*r + a1*r*r*r*r + r*r*r*r*r
    return [int(a1_new), int(a2_new), int(a3_new), int(a4_new), int(a6_new)]

def write_entry(f, details, row_id, group_id, task, rewrite, ainvs, rank):
    input_str = tokenize_ainvs(ainvs)
    label_str = str(rank)
    f.write(f"{input_str}\t{label_str}\n")
    details.writerow([row_id, group_id, task, rewrite, input_str, label_str])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_fraction", type=float, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    args = parser.parse_args()

    # Files
    train_f = open(f"{TASK_NAME}.train", "w")
    valid_f = open(f"{TASK_NAME}.valid", "w")
    test_f = open(f"{TASK_NAME}.test", "w")
    robust_f = open(f"{TASK_NAME}.robust", "w")
    details_f = open("details.csv", "w", newline='')
    details = csv.writer(details_f)
    details.writerow(["row_id", "group_id", "task", "rewrite", "input", "label"])

    # Estimate group count
    ex_per_group = args.rewrite_fraction * 3 + (1 - args.rewrite_fraction) * 1
    group_budget = int(args.num_examples / ex_per_group)

    print(f"Fetching ~{group_budget} isogeny classes...")
    class_labels = fetch_all_class_labels(max_classes=group_budget * 2)
    random.shuffle(class_labels)

    splits = (
        ["train"] * int(group_budget * TRAIN_RATIO)
        + ["valid"] * int(group_budget * VALID_RATIO)
        + ["test"]  * (group_budget - int(group_budget * TRAIN_RATIO) - int(group_budget * VALID_RATIO))
    )
    random.shuffle(splits)

    row_id, total = 0, 0
    for i, class_label in enumerate(class_labels):
        if total >= args.num_examples: break
        try:
            curves = fetch_class_curves(class_label)
        except Exception as e:
            print(f"Skipping {class_label}: {e}")
            continue
        if not curves: continue

        original = next((c for c in curves if c.get("lmfdb_number") == 1), curves[0])
        ainvs = original.get("ainvs")
        rank = original.get("rank")
        if ainvs is None or rank is None:
            continue

        split = splits[min(i, len(splits)-1)]
        fout = {"train": train_f, "valid": valid_f, "test": test_f}[split]
        group_id = f"ec-{i}"

        # Original
        write_entry(fout, details, row_id, group_id, TASK_NAME, "False", ainvs, rank)
        write_entry(robust_f, details, row_id, group_id, TASK_NAME, "False", ainvs, rank)
        row_id += 1
        total += 1

        if random.random() < args.rewrite_fraction:
            # One isogeny variant
            for curve in curves:
                if curve == original: continue
                ainvs2 = curve.get("ainvs")
                rank2 = curve.get("rank")
                if ainvs2 and rank2 is not None:
                    write_entry(robust_f, details, row_id, group_id, TASK_NAME, "ISOGENY", ainvs2, rank2)
                    row_id += 1
                    total += 1
                    break

            # One isomorphism
            ainvs_iso = generate_isomorphic_variant(ainvs)
            write_entry(robust_f, details, row_id, group_id, TASK_NAME, "ISOMORPHISM", ainvs_iso, rank)
            row_id += 1
            total += 1

    for f in [train_f, valid_f, test_f, robust_f, details_f]:
        f.close()
    print(f"Done. {total} examples written.")

if __name__ == "__main__":
    main()

