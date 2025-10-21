#!/usr/bin/env python3
import argparse, csv, os, random
from typing import List, Tuple, Dict

# ---------------- Sanitize helpers ----------------
SPACE = " "
SMALL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def tok(s: str) -> str:
    # Force plain ASCII spaces and strip weird whitespace (NBSP, CR, etc.)
    return s.replace("\u00A0", " ").replace("\r", "").strip()

def join(tokens: List[str]) -> str:
    # Join with single ASCII spaces, no empties
    return SPACE.join(t for t in (tok(t) for t in tokens) if t)

def emit_line(src: str, tgt: str, sep: str) -> str:
    # Force TAB delimiter or user-specified sep, no surrounding spaces
    return f"{tok(src)}{sep}{tok(tgt)}"

# ---------------- Tokenization ----------------
def _tok_int_positional(x: int, base: int) -> List[str]:
    sgn = "+" if x >= 0 else "-"
    x = abs(x)
    if x == 0:
        return [sgn, "0"]
    digs = []
    while x > 0:
        digs.append(str(x % base))
        x //= base
    return [sgn] + list(reversed(digs))

def _enc_scalar_signed(y: int, base_out: int) -> str:
    # Signed positional (default) -> works with PositionalInts on output
    return join(_tok_int_positional(y, base_out))

def _enc_scalar_unsigned(y: int) -> str:
    # Single-token unsigned label -> for SymbolicInts(max_class), if needed
    return str(abs(int(y)))

def _enc_gcd_src(a: int, b: int, base_in: int) -> str:
    # Add a separator between the two numbers that Int2Int can understand
    a_tokens = _tok_int_positional(a, base_in)
    b_tokens = _tok_int_positional(b, base_in)
    # Use a special separator token like "GCD" or "|"
    return join(a_tokens+ b_tokens)

def _enc_modexp_src(a: int, b: int, n: int, base_in: int) -> str:
    return join(_tok_int_positional(a, base_in) +
                _tok_int_positional(b, base_in) +
                _tok_int_positional(n, base_in))

def _enc_ec_src(coeffs: Tuple[int,int,int,int,int], base_in: int) -> str:
    toks: List[str] = []
    for v in coeffs:
        toks += _tok_int_positional(v, base_in)
    return join(toks)

def _enc_legendre_src(a: int, p: int, base_in: int) -> str:
    return join(_tok_int_positional(a, base_in) +
                _tok_int_positional(p, base_in))

# ---------------- Oracles ----------------
def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _modexp(a: int, b: int, n: int) -> int:
    b = max(0, b)
    return pow(a % n, b, n)

def _legendre(a: int, p: int) -> int:
    answer = pow(a % p, (p - 1) // 2, p)
    if answer == p-1:
        return -1
    return answer   

def _log_mod_prime(a: int, b: int, p: int) -> int:
    """
    Find x such that a^x = b mod p, where p is prime.
    Return -1 if no such x exists. (Brute force; only for small p.)
    """
    a = a % p; b = b % p
    if b == 1:
        return 0
    cur = a
    for x in range(1, p):
        if cur == b:
            return x
        cur = (cur * a) % p
    return -1

# ---------------- Robustness transforms (SPR) ----------------
def rw_gcd_comm(a,b): return (b,a), "COMM"
def rw_gcd_sign(a,b): return (abs(a),abs(b)), "SIGN"
def rw_gcd_bez_a(a,b,k): return (a + k*b, b), f"BEZ_a(k={k})"
def rw_gcd_bez_b(a,b,k): return (a, b + k*a), f"BEZ_b(k={k})"

def rw_modexp_base_lift(a,b,n,k): return (a + k*n, b, n), f"BASE_LIFT(k={k})"
def rw_modexp_pow_factor(a,b,n,k):
    assert k >= 2 and (b % k == 0)
    a_prime = pow(a % n, k, n)
    return (a_prime, b // k, n), f"POW_FACTOR(k={k})"

def rw_legendre_add(a,p,k): return (a + k*p, p), f"ADD(k={k})"
def rw_legendre_mul_ps(a,p,k): return (a * pow(k,2,p), p), f"MUL_PS(k={k})"

# ---------------- IO ----------------
def _write_lines(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline='\n' guarantees LF; encoding='utf-8' is standard
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            # sanitize again just in case
            f.write(tok(s) + "\n")

def _write_details_csv(path: str, rows: List[Dict]):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# ---------------- Generators (grouped for RSP) ----------------
def gen_groups_gcd(num_groups, base_in, base_out, a_min, a_max, rewrites_per, k_choices, seed, unsigned_targets):
    rng = random.Random(seed)
    groups = []
    enc_tgt = (lambda y: _enc_scalar_unsigned(y)) if unsigned_targets else (lambda y: _enc_scalar_signed(y, base_out))
    for gid in range(num_groups):
        a = rng.randint(a_min, a_max); b = rng.randint(a_min, a_max)
        y = _gcd(a,b); label = enc_tgt(y)
        variants = [("ORIG", _enc_gcd_src(a,b,base_in), label)]
        for _ in range(rewrites_per):
            k = rng.choice(k_choices)
            if rng.random() < 0.5: (aa,bb), nm = rw_gcd_bez_a(a,b,k)
            else:                  (aa,bb), nm = rw_gcd_bez_b(a,b,k)
            variants.append((nm, _enc_gcd_src(aa,bb,base_in), label))
        (aa,bb), nm = rw_gcd_comm(a,b); variants.append((nm, _enc_gcd_src(aa,bb,base_in), label))
        (aa,bb), nm = rw_gcd_sign(a,b); variants.append((nm, _enc_gcd_src(aa,bb,base_in), label))
        groups.append((f"gcd-{gid}", variants))
    return groups

def gen_groups_modexp(num_groups, base_in, base_out, a_min,a_max,b_min,b_max,n_min,n_max, rewrites_per, k_choices, seed, unsigned_targets):
    rng = random.Random(seed)
    enc_tgt = (lambda y: _enc_scalar_unsigned(y)) if unsigned_targets else (lambda y: _enc_scalar_signed(y, base_out))
    groups = []
    for gid in range(num_groups):
        a = rng.randint(a_min, a_max)
        b = rng.randint(b_min, b_max)
        n = max(2, rng.randint(n_min, n_max))
        y = _modexp(a,b,n); label = enc_tgt(y)
        variants = [("ORIG", _enc_modexp_src(a,b,n,base_in), label)]
        for _ in range(rewrites_per):
            if rng.random() < 0.5:
                k = rng.choice(k_choices)
                (aa,bb,nn), nm = rw_modexp_base_lift(a,b,n,k)
                variants.append((nm, _enc_modexp_src(aa,bb,nn,base_in), label))
            else:
                ks = [kk for kk in k_choices if kk >= 2 and (b % kk == 0)]
                if ks:
                    k = rng.choice(ks)
                    (aa,bb,nn), nm = rw_modexp_pow_factor(a,b,n,k)
                    variants.append((nm, _enc_modexp_src(aa,bb,nn,base_in), label))
        groups.append((f"modexp-{gid}", variants))
    return groups

def gen_groups_ec_rank_from_csv(pairs_csv, base_in, base_out, seed, unsigned_targets):
    rng = random.Random(seed)
    enc_tgt = (lambda y: _enc_scalar_unsigned(y)) if unsigned_targets else (lambda y: _enc_scalar_signed(y, base_out))
    by_group: Dict[str, List[Dict[str,str]]] = {}
    with open(pairs_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            by_group.setdefault(r["group_id"], []).append(r)
    groups, gid_ctr = [], 0
    for _, reps in by_group.items():
        if len(reps) < 2: continue
        reps = reps[:]; rng.shuffle(reps)
        m = len(reps)
        for i, r in enumerate(reps):
            a1,a2,a3,a4,a6 = map(int, (r["a1"], r["a2"], r["a3"], r["a4"], r["a6"]))
            rank = int(r["rank"])
            src0 = _enc_ec_src((a1,a2,a3,a4,a6), base_in)
            r2 = reps[(i+1) % m]
            a1p,a2p,a3p,a4p,a6p = map(int, (r2["a1"], r2["a2"], r2["a3"], r2["a4"], r2["a6"]))
            src1 = _enc_ec_src((a1p,a2p,a3p,a4p,a6p), base_in)
            label = enc_tgt(rank)
            groups.append((f"ec-{gid_ctr}", [("ORIG", src0, label), ("ISOG_REP", src1, label)]))
            gid_ctr += 1
    return groups

def gen_groups_legendre(num_groups, base_in, base_out, a_min, a_max, p_min, p_max, rewrites_per, k_choices, seed, unsigned_targets):
    rng = random.Random(seed)
    groups = []
    enc_tgt = (lambda y: _enc_scalar_unsigned(y)) if unsigned_targets else (lambda y: _enc_scalar_signed(y, base_out))
    for gid in range(num_groups):
        a = rng.randint(a_min, a_max)
        # Ensure p is an odd prime
        p = rng.choice([p for p in SMALL_PRIMES if p_min <= p <= p_max])
        y = _legendre(a,p); label = enc_tgt(y)
        variants = [("ORIG", _enc_legendre_src(a,p,base_in), label)]
        for _ in range(rewrites_per):
            if rng.random() < 0.5:
                k = rng.choice(k_choices)
                (aa,pp), nm = rw_legendre_add(a,p,k)
                variants.append((nm, _enc_legendre_src(aa,pp,base_in), label))
            else:
                k = rng.choice([kk for kk in k_choices if kk % p != 0])
                (aa,pp), nm = rw_legendre_mul_ps(a,p,k)
                variants.append((nm, _enc_legendre_src(aa,pp,base_in), label))
        groups.append((f"legendre-{gid}", variants))
    return groups

# ---------------- Emit ----------------
def _emit(task: str, groups, out_dir: str, train_frac: float, valid_frac: float, sep: str):
    task_dir = os.path.join(out_dir, task); os.makedirs(task_dir, exist_ok=True)
    random.shuffle(groups)
    n = len(groups); n_tr = int(train_frac*n); n_va = int(valid_frac*n)
    train_g, valid_g, test_g = groups[:n_tr], groups[n_tr:n_tr+n_va], groups[n_tr+n_va:]

    def only_origs(group_slice):
        return [emit_line(src, tgt, sep) for _, variants in group_slice for name, src, tgt in variants if name=="ORIG"]

    def robust_all(groups_all):
        lines, rows = [], []; rid = 0
        for gid, variants in groups_all:
            for name, src, tgt in variants:
                lines.append(emit_line(src, tgt, sep))
                rows.append({"row_id": rid, "group_id": gid, "task": task, "rewrite": name, "input": tok(src), "label": tok(tgt)})
                rid += 1
        return lines, rows

    _write_lines(os.path.join(task_dir, f"{task}.train"), only_origs(train_g))
    _write_lines(os.path.join(task_dir, f"{task}.valid"), only_origs(valid_g))
    _write_lines(os.path.join(task_dir, f"{task}.test"),  only_origs(test_g))
    robust_lines, details = robust_all(groups)
    _write_lines(os.path.join(task_dir, f"{task}.robust"), robust_lines)
    _write_details_csv(os.path.join(task_dir, "details.csv"), details)

    print(f"[OK] {task}: train={len(only_origs(train_g))} valid={len(only_origs(valid_g))} "
          f"test={len(only_origs(test_g))} robust={len(robust_lines)} -> {task_dir}")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gcd","modexp","ec_rank", "legendre"], required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--base_in", type=int, default=10)
    ap.add_argument("--base_out", type=int, default=10)  # keep targets base-10 (safer)
    ap.add_argument("--unsigned_targets", action="store_true", help="Emit single-token unsigned targets (for SymbolicInts)")

    ap.add_argument("--num_groups", type=int, default=220000)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--valid_frac", type=float, default=0.1)

    ap.add_argument("--rewrites_per", type=int, default=4)
    ap.add_argument("--k_choices", type=int, nargs="+", default=[-3,-2,-1,1,2,3])

    ap.add_argument("--a_min", type=int, default=-10**6)
    ap.add_argument("--a_max", type=int, default= 10**6)
    ap.add_argument("--b_min", type=int, default=0)
    ap.add_argument("--b_max", type=int, default=256)
    ap.add_argument("--n_min", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=10**6)

    ap.add_argument("--ec_pairs_csv", type=str, default=None)

    ap.add_argument("--sep", type=str, default="\t", help="input/target delimiter (paper uses TAB)")

    args = ap.parse_args()
    random.seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)

    if args.task == "gcd":
        groups = gen_groups_gcd(args.num_groups, args.base_in, args.base_out,
                                args.a_min, args.a_max, args.rewrites_per, args.k_choices, args.seed,
                                args.unsigned_targets == True)
    elif args.task == "modexp":
        groups = gen_groups_modexp(args.num_groups, args.base_in, args.base_out,
                                   args.a_min, args.a_max, args.b_min, args.b_max, args.n_min, args.n_max,
                                   args.rewrites_per, args.k_choices, args.seed,
                                   args.unsigned_targets == True)
        
    elif args.task == "legendre":
        groups = gen_groups_legendre(args.num_groups, args.base_in, args.base_out,
                                   args.a_min, args.a_max, args.n_min, args.n_max,
                                   args.rewrites_per, args.k_choices, args.seed,
                                   args.unsigned_targets == True)
    else:
        if not args.ec_pairs_csv:
            raise SystemExit("--task ec_rank requires --ec_pairs_csv")
        groups = gen_groups_ec_rank_from_csv(args.ec_pairs_csv, base_in=args.base_in, base_out=args.base_out,
                                             seed=args.seed, unsigned_targets=args.unsigned_targets == True)

    _emit(args.task, groups, args.out_dir, args.train_frac, args.valid_frac, args.sep)

if __name__ == "__main__":
    main()
