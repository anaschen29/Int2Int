# mix_train.py
import sys, os, random, pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: python mix_train.py <DATA_DIR> <p1> [<p2> ...]")
        sys.exit(1)

    root = sys.argv[1].rstrip("/")
    probs = [float(x) for x in sys.argv[2:]]
    random.seed(1337)

    details_csv = os.path.join(root, "details.csv")
    train_path  = os.path.join(root, "gcd.train")
    assert os.path.isfile(details_csv), f"not found: {details_csv}"
    assert os.path.isfile(train_path),  f"not found: {train_path}"

    details = pd.read_csv(details_csv)

    # IMPORTANT: read only the INPUT side from gcd.train (split on TAB)
    orig_inputs = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src = parts[0].strip()
            orig_inputs.add(src)

    # Find which groups were used for training (via input match)
    # (details['input'] is exactly the src string)
    df_tr = details[details["input"].isin(orig_inputs)].copy()

    # Safety: bail clearly if nothing matched
    if df_tr.empty:
        print("No matches between gcd.train inputs and details.csv 'input'.")
        print("Debug counts:",
              f"  train unique inputs={len(orig_inputs)}",
              f"  details rows={len(details)}",
              f"  details unique inputs={details['input'].nunique()}",
              sep="\n")
        sys.exit(2)

    for p in probs:
        out_lines = []
        for gid, g in df_tr.groupby("group_id"):
            g = g.sample(frac=1.0, random_state=1337)
            orig = g[g.rewrite == "ORIG"].iloc[0]
            # With prob p, replace ORIG by a random rewrite from the same group
            if random.random() < p:
                rew = g[g.rewrite != "ORIG"]
                row = rew.sample(1, random_state=1337).iloc[0] if len(rew) else orig
            else:
                row = orig
            out_lines.append(f"{row['input']}\t{row['label']}")

        of = os.path.join(root, f"gcd.train.mixed_p{int(round(p*100))}")
        with open(of, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"Wrote {of}  ({len(out_lines)} lines)")

if __name__ == "__main__":
    main()
