"""
Requests elliptic curve data from an LMFDB SQL mirror using lmfdb-lite. 

USAGE : 
Step 1: Ensure lmfdb-lite is installed and configured with DB credentials.
    pip install -U "lmfdb-lite[pgbinary] @ git+https://github.com/roed314/lmfdb-lite.git"
Step 2: Open a terminal and initialize credentials:
    setx LMFDB_HOST "devmirror.lmfdb.xyz"
    setx LMFDB_NAME "lmfdb"
    setx LMFDB_USER "lmfbd"
    setx LMFDB_PASSWORD "lmfdb"
    setx LMFDB_PORT "5432"
Step 3: Run this script in terminal to build the dataset: 
    python request_ec_data.py --total 10000 --rewrite_pct 0.5 --outdir ec_rank_dataset --seed 1337 

    seed and outdir are optional (default seed=1337, outdir=ec_rank_dataset).
"""

import argparse, csv, os, random, sys, math, re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# ------------- LMFDB-lite (SQL mirror) -----------------
# Install:  pip3 install -U "lmfdb-lite[pgbinary] @ git+https://github.com/roed314/lmfdb-lite.git"
# Usage examples & available columns: db.ec_curvedata.search(..., ["lmfdb_label","conductor","ainvs","rank"])  (see README) 
# Source: roed314/lmfdb-lite (README usage/snippets)  ← cites at end
try:
    from lmf import db
except Exception as e:
    print("ERROR: Could not import lmfdb-lite. Install it and ensure DB credentials for the LMFDB SQL mirror are set.", file=sys.stderr)
    print("       pip3 install -U \"lmfdb-lite[pgbinary] @ git+https://github.com/roed314/lmfdb-lite.git\"", file=sys.stderr)
    raise

# ---------------- Formatting helpers -------------------

def tok_int(n: int) -> str:
    """
    Always print an explicit sign and space-separated digits, including zero.
    Examples:
      -123 -> '- 1 2 3'
       +81 -> '+ 8 1'
        0  -> '+ 0'
    """
    sign = "+" if n >= 0 else "-"
    return f"{sign} " + " ".join(str(abs(n)))


def tok_ainvs(ainvs: List[int]) -> str:
    """
    Tokenize the 5 Weierstrass coefficients [a1,a2,a3,a4,a6]
    like '+ 1 + 0 + 0 + 3 + 8 1', preserving gcd-style spacing.
    """
    return " ".join(tok_int(a) for a in ainvs)

def make_group_id(seed_label: str, counter: int) -> str:
    return f"ec-{seed_label.replace('.', '_')}-{counter}"

def label_from_lmfdb_label(l: str) -> Tuple[int, str, int]:
    """
    Parse 'lmfdb_label' like '37.a1' -> (conductor=37, iso_letter='a', number=1).
    This is the standard format in ec_curvedata.
    """
    # Robust parse: N.classletter + number
    m = re.fullmatch(r"(\d+)\.([a-z]+)(\d+)", l)
    if not m:
        # try another common style like '121.b3' etc.
        raise ValueError(f"Unexpected elliptic curve label format: {l}")
    N, iso_letter, num = int(m.group(1)), m.group(2), int(m.group(3))
    return N, iso_letter, num

def isoclass_prefix(l: str) -> str:
    """'37.a1' -> '37.a'."""
    N, iso, _ = label_from_lmfdb_label(l)
    return f"{N}.{iso}"

# --------------- Isomorphism rewrite -------------------
# General integral change (over Q) on a long Weierstrass model:
# (x, y) = (u^2 x' + r, u^3 y' + s u^2 x' + t) with u=±1 (keep integral), r,s,t ∈ Z.
# This preserves isomorphism class and hence the rank. For small r,s,t we stay integral.
# We implement the coefficient transformation for (a1,a2,a3,a4,a6) when u=±1.
#
# Reference formulas (with u=±1):
# a1' = a1 + 2*s
# a2' = a2 - s*a1 + 3*r - s*s
# a3' = a3 + r*a1 + 2*t
# a4' = a4 - s*a3 + 2*r*a2 - (t + r*s)*a1 + 3*r*r - 2*s*t
# a6' = a6 + r*a4 + r*r*a2 + r*r*r - t*a3 - r*t*a1 - t*t
#
# (Same as Sage/LMFDB model transform specialized to u=±1 to keep integrality simple.)

def isomorphic_ainvs(a1,a2,a3,a4,a6, r:int, s:int, t:int, u:int=1) -> List[int]:
    if u not in (1,-1):
        raise ValueError("This simplified transform only supports u=±1.")
    # CHANGED: incorporate u^{-k} scaling. For u=-1, only a1' and a3' flip sign.
    a1_num = a1 + 2*s
    a2_num = a2 - s*a1 + 3*r - s*s
    a3_num = a3 + r*a1 + 2*t
    a4_num = a4 - s*a3 + 2*r*a2 - (t + r*s)*a1 + 3*r*r - 2*s*t
    a6_num = a6 + r*a4 + r*r*a2 + r*r*r - t*a3 - r*t*a1 - t*t

    a1p =  a1_num if u == 1 else -a1_num   # CHANGED
    a2p =  a2_num                           # CHANGED (u^2 = 1)
    a3p =  a3_num if u == 1 else -a3_num   # CHANGED
    a4p =  a4_num                           # CHANGED (u^4 = 1)
    a6p =  a6_num                           # CHANGED (u^6 = 1)
    return [a1p, a2p, a3p, a4p, a6p]

def sample_isomorphism(ainvs: List[int]) -> Tuple[List[int], str]:
    # Small moves keep coefficients from blowing up; tune as you like.
    r = random.randint(-2,2)
    s = random.randint(-2,2)
    t = random.randint(-2,2)
    u = random.choice([1,-1])
    new_ainvs = isomorphic_ainvs(*ainvs, r=r, s=s, t=t, u=u)
    rewrite_tag = f"ISOM(u={u},r={r},s={s},t={t})"
    return new_ainvs, rewrite_tag

# --------------- Isogeny rewrite -----------------------

def collect_isogenous(db_cache_by_iso: Dict[str, List[Tuple[str,List[int],int]]],
                      conductor: int, iso_letter: str) -> List[Tuple[str,List[int],int]]:
    """
    Get all curves in the given isogeny class, projected as (lmfdb_label, ainvs, rank).
    Uses a small cache so we don't re-query the same class a million times.
    """
    key = f"{conductor}.{iso_letter}"
    if key in db_cache_by_iso:
        return db_cache_by_iso[key]

    # Strategy: search by conductor, then filter by iso_letter from the label.
    # The ec_curvedata table exposes 'lmfdb_label', 'conductor', 'ainvs', 'rank' (README).
    rows = list(db.ec_curvedata.search({"conductor": conductor}, ["lmfdb_label","ainvs","rank"], limit=5000))
    out = []
    for row in rows:
        lbl = row["lmfdb_label"]
        try:
            N, iso, _ = label_from_lmfdb_label(lbl)
        except ValueError:
            continue
        if N == conductor and iso == iso_letter:
            out.append((lbl, row["ainvs"], row["rank"]))
    # Sort by number index for stability
    def num_part(lbl:str) -> int:
        return label_from_lmfdb_label(lbl)[2]
    out.sort(key=lambda t: num_part(t[0]))
    db_cache_by_iso[key] = out
    return out

# ---------------- Dataset assembly ---------------------

def write_split_files(outdir: str, splits: Dict[str, List[Tuple[str,int,str,str]]]):
    """
    splits[splitname] = list of (input_str, label_int, group_id, rewrite_tag)
    Writes split .txt files and a details.csv capturing provenance.
    """
    os.makedirs(outdir, exist_ok=True)
    details_path = os.path.join(outdir, "details.csv")
    fieldnames = ["row_id","group_id","task","rewrite","input","label"]
    row_id = 0
    with open(details_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for split_name, rows in splits.items():
            with open(os.path.join(outdir, f"{split_name}.txt"), "w") as fout:
                for (inp, lab, gid, rtag) in rows:
                    # task name fixed:
                    w.writerow({
                        "row_id": row_id,
                        "group_id": gid,
                        "task": "ec_rank",
                        "rewrite": rtag,
                        "input": inp,
                        "label": str(lab),
                    })
                    row_id += 1
                    fout.write(f"{inp}\t{tok_int(int(lab))}\n")

def do_splits(examples: List[Tuple[str,int,str,str]]) -> Dict[str, List[Tuple[str,int,str,str]]]:
    """
    Train/valid/test/robust split.
    Ensure TRAIN and TEST are (as much as possible) balanced between rank 0 and rank 1,
    and make sure each split is shuffled.
    """
    n = len(examples)
    n_train = int(0.40 * n)
    n_valid = int(0.05 * n)
    n_test  = int(0.05 * n)
    assigned = n_train + n_valid + n_test
    n_robust = max(0, n - assigned)

    # Separate by label and shuffle class buckets to avoid label blocks
    ex0 = [e for e in examples if e[1] == 0]
    ex1 = [e for e in examples if e[1] == 1]
    random.shuffle(ex0)
    random.shuffle(ex1)

    def take_balanced(n_take: int, ex0: List, ex1: List) -> List:
        half = n_take // 2
        out = []

        t0 = min(len(ex0), half)
        out.extend(ex0[:t0]); del ex0[:t0]

        t1 = min(len(ex1), n_take - len(out))
        out.extend(ex1[:t1]); del ex1[:t1]

        # If one side was short, top up from the other
        rem = n_take - len(out)
        if rem > 0 and len(ex0) > 0:
            t = min(len(ex0), rem)
            out.extend(ex0[:t]); del ex0[:t]; rem -= t
        if rem > 0 and len(ex1) > 0:
            t = min(len(ex1), rem)
            out.extend(ex1[:t]); del ex1[:t]

        random.shuffle(out)  # ensure the split itself is shuffled
        return out

    train = take_balanced(n_train, ex0, ex1)
    test  = take_balanced(n_test,  ex0, ex1)

    remaining = ex0 + ex1
    random.shuffle(remaining)  # shuffle before slicing the rest
    valid = remaining[:n_valid]
    robust = remaining[n_valid:n_valid + n_robust]

    # Extra safety: shuffle these small splits too
    random.shuffle(valid)
    random.shuffle(robust)

    return {
        "train": train,
        "valid": valid,
        "test":  test,
        "robust": robust,
    }


# ---------------- Main builder -------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools, time

def main():
    ap = argparse.ArgumentParser(description="Fast EC-rank dataset builder.")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--rewrite_pct", type=float, required=True)
    ap.add_argument("--outdir", type=str, default="ec_rank_dataset_fast")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of threads for rewrite generation.")
    args = ap.parse_args()

    random.seed(args.seed)

    N = args.total
    rho = args.rewrite_pct
    avg_k = rho / (1 - rho) if rho < 1 else 2
    min_k, max_k = int(avg_k // 1), max(1, int(avg_k // 1) + 1)
    rew_alt = True

    print(f"Fetching {N} base curves from LMFDB mirror...", flush=True)
    t0 = time.time()
    # LIMIT CONDUCTOR TO <= 1e5
    seeds_iter = db.ec_curvedata.search(
        {"conductor": {"$lte": 100000}},
        ["lmfdb_label","ainvs","rank"],
        limit=N
    )
    seeds = list(itertools.islice(seeds_iter, N))
    print(f"Fetched {len(seeds)} seeds in {time.time()-t0:.1f}s")

    examples = []
    iso_cache = {}

    def process_seed(row):
        nonlocal rew_alt
        lbl, ainvs, rank = row["lmfdb_label"], row["ainvs"], int(row["rank"])
        if rank not in (0, 1):
            return []  # CHANGED: restrict dataset to rank 0 or 1 curves only
        if not isinstance(ainvs, list) or len(ainvs)!=5:
            return []
        gid = make_group_id(lbl, random.randint(1, 1_000_000_000))
        rows = [(tok_ainvs(ainvs), rank, gid, "ORIG")]
        k = min_k if rew_alt else max_k
        rew_alt = not rew_alt

        # --- all rewrites are isomorphisms only (skip slow isogeny DB calls) ---
        k_iso = k      # all rewrites
        for _ in range(k_iso):
            new_ainvs, tag = sample_isomorphism(ainvs)
            rows.append((tok_ainvs(new_ainvs), rank, gid, tag))

        return rows

    print(f"Generating rewrites using {args.workers} threads...")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_seed, s) for s in seeds]
        for i, fut in enumerate(as_completed(futs)):
            examples.extend(fut.result())
            if i % 2000 == 0:
                print(f"{i}/{len(seeds)} seeds done...", flush=True)
    print(f"Rewrites done in {time.time()-t1:.1f}s")

    random.shuffle(examples)
    splits = do_splits(examples)
    write_split_files(args.outdir, splits)
    print(f"Wrote {sum(len(v) for v in splits.values())} examples to {args.outdir}")

if __name__ == "__main__":
    main()