# src/metadata_eval.py
from typing import Any, Dict, List, Optional
import os
import pandas as pd
import torch

def evaluate_with_metadata(
    *,
    modules: Dict[str, Any],
    env: Any,
    params: Any,
    epoch: int,
    metadata_path: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate using the Int2Int encoder/decoder API faithfully, then join with metadata.
    Produces <output_dir>/<split>/labels_{epoch}.csv with metadata columns + 'prediction'.
    Matching is done on a whitespace-normalized 'input' key to be robust to spacing.
    """

    # ---------- IO setup ----------
    if output_dir is None:
        output_dir = params.dump_path
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Metadata load & canonical key ----------
    def canon(s: str) -> str:
        # collapse any whitespace runs to single spaces and strip ends
        # (safe because your data is already sign/digit tokens with spaces)
        return " ".join(str(s).split())

    meta_df = pd.read_csv(metadata_path)
    if "input" not in meta_df.columns:
        raise ValueError("metadata CSV must contain an 'input' column.")
    meta_df["_key"] = meta_df["input"].apply(canon)
    meta_map = meta_df.set_index("_key")
    # quick visibility
    print(f"[metadata_eval] loaded metadata rows={len(meta_df)} cols={list(meta_df.columns)}", flush=True)

    # ---------- Model handles ----------
    arch = params.architecture
    is_multi_gpu = getattr(params, "multi_gpu", False)

    encoder = modules.get("encoder", None)
    decoder = modules.get("decoder", None)
    if is_multi_gpu and encoder is not None:
        encoder = encoder.module
    if is_multi_gpu and decoder is not None:
        decoder = decoder.module

    # ---------- Eval data paths (splits) ----------
    if getattr(params, "eval_data", ""):
        data_paths = [p for p in params.eval_data.split(",") if p]
    else:
        data_paths = [None]

    # small helpers
    def tokens_to_str(ids: List[int]) -> str:
        return " ".join(env.id2word[t] for t in ids)

    SPECIAL = {
        env.eos_index,
        getattr(env, "bos_index", -1),
        getattr(env, "sep_index", -1),
        env.pad_index,
    }

    max_len = int(getattr(params, "max_output_len", 128))
    if max_len <= 0:
        max_len = 128
    max_len = max_len + 2  # allow BOS/EOS

    # ---------- Tasks list ----------
    if isinstance(params.tasks, (list, tuple)):
        task_list = list(params.tasks)
    else:
        task_list = [t for t in str(params.tasks).split(",") if t]

    all_results: List[Dict[str, Any]] = []  # aggregated over all splits (also return)

    for data_path in data_paths:
        split_name = os.path.basename(data_path) if data_path else "unknown"
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"[metadata_eval] evaluating split={split_name} -> out={split_dir}", flush=True)

        results: List[Dict[str, Any]] = []
        total_seen = 0
        matched = 0
        skipped = 0

        path_list = [data_path] if data_path is not None else None

        for task in task_list:
            iterator = env.create_test_iterator(
                data_type="valid",            # evaluation mode
                task=task,
                data_path=path_list,
                batch_size=params.batch_size_eval,
                params=params,
                size=params.eval_size,
            )

            # eval mode
            if encoder is not None:
                encoder.eval()
            if decoder is not None:
                decoder.eval()

            with torch.no_grad():
                for (x1, len1), (x2, len2), _ in iterator:
                    # to CUDA if available
                    if torch.cuda.is_available() and not getattr(params, "cpu", False):
                        x1, len1 = x1.cuda(non_blocking=True), len1.cuda(non_blocking=True)

                    bs = len1.size(0)

                    # ---------- Reconstruct inputs (drop trailing EOS) ----------
                    inp_strs: List[str] = []
                    for i in range(bs):
                        L = int(len1[i].item())    # includes EOS added by collator
                        use_L = max(0, L - 1)      # drop trailing EOS to match file/CSV
                        ids = x1[:use_L, i].detach().cpu().tolist()
                        ids = [t for t in ids if t not in SPECIAL]  # defensive
                        inp_strs.append(tokens_to_str(ids))

                    # ---------- Generate predictions ----------
                    pred_strs: List[str] = [""] * bs

                    if arch == "encoder_decoder":
                        enc_states = encoder(
                            "fwd",
                            x=x1,
                            lengths=len1,
                            causal=False,
                            src_enc=None,
                            src_len=None,
                            use_cache=False,
                        )  # (slen_src, bs, dim)
                        src_enc = enc_states.transpose(0, 1).contiguous()  # (bs, slen, dim)
                        gen_tokens, gen_len = decoder.generate(src_enc, len1, max_len=max_len)  # (cur_len, bs)

                        gen_tokens = gen_tokens.cpu()
                        for i in range(bs):
                            seq = gen_tokens[:, i].tolist()
                            # drop BOS at 0; stop at first EOS >=1
                            eos_pos = next((j for j in range(1, len(seq)) if seq[j] == env.eos_index), len(seq))
                            body = [t for t in seq[1:eos_pos] if t not in SPECIAL]
                            pred_strs[i] = tokens_to_str(body)

                    elif arch == "decoder_only":
                        gen_tokens, gen_len = decoder.generate(x1, len1, max_len=max_len)  # (cur_len, bs)
                        gen_tokens = gen_tokens.cpu()
                        for i in range(bs):
                            seq = gen_tokens[:, i].tolist()
                            eos_pos = next((j for j in range(1, len(seq)) if seq[j] == env.eos_index), len(seq))
                            body = [t for t in seq[1:eos_pos] if t not in SPECIAL]
                            pred_strs[i] = tokens_to_str(body)

                    elif arch == "encoder_only":
                        if hasattr(encoder, "decode"):
                            preds = encoder.decode(x1, len1, exp_len=max_len).cpu()  # (bs, <=max_len)
                            for i in range(bs):
                                seq = preds[i].tolist()
                                eos_pos = next((j for j, t in enumerate(seq) if t == env.eos_index), len(seq))
                                body = [t for t in seq[:eos_pos] if t not in SPECIAL]
                                pred_strs[i] = tokens_to_str(body)
                        else:
                            enc_states = encoder(
                                "fwd", x=x1, lengths=len1, causal=False, src_enc=None, src_len=None, use_cache=False
                            )
                            last = enc_states[-1]                   # (bs, dim)
                            scores = encoder.proj(last)             # (bs, n_words)
                            next_words = torch.topk(scores, 1)[1].squeeze(1).cpu().tolist()
                            pred_strs = [env.id2word[w] if w not in SPECIAL else "" for w in next_words]

                    # ---------- Join with metadata (whitespace-normalized key) ----------
                    for i in range(bs):
                        total_seen += 1
                        key = canon(inp_strs[i])
                        pred = pred_strs[i]
                        try:
                            rows = meta_map.loc[key]
                        except KeyError:
                            skipped += 1
                            continue
                        matched += 1
                        if isinstance(rows, pd.DataFrame):
                            recs = rows.to_dict("records")
                        else:
                            recs = [rows.to_dict()]
                        for rec in recs:
                            out = dict(rec)
                            out["prediction"] = pred
                            results.append(out)

        # ---------- Write one CSV per split ----------
        out_df = pd.DataFrame(results)
        csv_path = os.path.join(split_dir, f"labels_{epoch}.csv")
        print(f"[metadata_eval] split={split_name} total_seen={total_seen} matched={matched} skipped={skipped}", flush=True)
        print(f"[metadata_eval] writing: {csv_path} (rows={len(out_df)})", flush=True)
        out_df.to_csv(csv_path, index=False)

        all_results.extend(results)

    # Return a DataFrame aggregating all splits (if multiple)
    return pd.DataFrame(all_results)
