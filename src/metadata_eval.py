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
    Produces labels_{epoch}.csv with columns from metadata plus 'prediction'.
    """

    # ---------- IO setup ----------
    if output_dir is None:
        output_dir = params.dump_path
    os.makedirs(output_dir, exist_ok=True)

    meta_df = pd.read_csv(metadata_path)
    if "input" not in meta_df.columns:
        raise ValueError("metadata CSV must contain an 'input' column.")
    # speed up lookup
    meta_map = meta_df.set_index("input")

    # ---------- Model handles ----------
    arch = params.architecture
    is_multi_gpu = getattr(params, "multi_gpu", False)

    encoder = modules.get("encoder", None)
    decoder = modules.get("decoder", None)
    if is_multi_gpu and encoder is not None:
        encoder = encoder.module
    if is_multi_gpu and decoder is not None:
        decoder = decoder.module

    # ---------- Eval data paths ----------
    if getattr(params, "eval_data", ""):
        data_paths = params.eval_data.split(",")
    else:
        data_paths = [None]

    results: List[Dict[str, Any]] = []

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
    # allow for BOS/EOS positions
    max_len = max_len + 2

    # ---------- Iterate over datasets / tasks ----------
    if isinstance(params.tasks, (list, tuple)):
        task_list = list(params.tasks)
    else:
        task_list = [t for t in str(params.tasks).split(",") if t]

    for data_path in data_paths:
        path_list = [data_path] if data_path is not None else None

        for task in task_list:
            iterator = env.create_test_iterator(
                data_type="valid",
                task=task,
                data_path=path_list,
                batch_size=params.batch_size_eval,
                params=params,
                size=params.eval_size,
            )

            # Put modules into eval mode
            if encoder is not None:
                encoder.eval()
            if decoder is not None:
                decoder.eval()

            with torch.no_grad():
                for (x1, len1), (x2, len2), _ in iterator:
                    # Send to CUDA if needed
                    if torch.cuda.is_available() and not getattr(params, "cpu", False):
                        x1, len1 = x1.cuda(non_blocking=True), len1.cuda(non_blocking=True)

                    bs = len1.size(0)

                    # ---------- Reconstruct input string exactly like in files (no trailing EOS) ----------
                    inp_strs: List[str] = []
                    for i in range(bs):
                        L = int(len1[i].item())  # includes EOS the collator added
                        use_L = max(0, L - 1)    # drop trailing EOS to match file/CSV
                        ids = x1[:use_L, i].detach().cpu().tolist()
                        # defensively drop any specials (rare)
                        ids = [t for t in ids if t not in SPECIAL]
                        inp_strs.append(tokens_to_str(ids))

                    # ---------- Generate predictions per architecture ----------
                    pred_strs: List[str] = [""] * bs

                    if arch == "encoder_decoder":
                        # 1) Encode with correct API
                        # encoder forward expects: encoder("fwd", x= (slen, bs), lengths=(bs), causal=False)
                        enc_states = encoder(
                            "fwd",
                            x=x1,
                            lengths=len1,
                            causal=False,
                            src_enc=None,
                            src_len=None,
                            use_cache=False,
                        )  # (slen_src, bs, dim)

                        # 2) decoder.generate expects src_enc (bs, slen_src, dim) and src_len (bs)
                        src_enc = enc_states.transpose(0, 1).contiguous()  # (bs, slen, dim)

                        gen_tokens, gen_len = decoder.generate(
                            src_enc, len1, max_len=max_len
                        )  # gen_tokens: (cur_len, bs)

                        # 3) Decode: sequence starts at BOS (<eos>) at pos 0; stop before first EOS
                        gen_tokens = gen_tokens.cpu()
                        for i in range(bs):
                            seq = gen_tokens[:, i].tolist()
                            # find first EOS at position >= 1
                            eos_pos = None
                            for j in range(1, len(seq)):
                                if seq[j] == env.eos_index:
                                    eos_pos = j
                                    break
                            if eos_pos is None:
                                eos_pos = len(seq)
                            # drop BOS at [0], drop EOS; also drop specials defensively
                            body = [t for t in seq[1:eos_pos] if t not in SPECIAL]
                            pred_strs[i] = tokens_to_str(body)

                    elif arch == "decoder_only":
                        # decoder_only.generate expects the input tokens and lengths directly
                        gen_tokens, gen_len = decoder.generate(
                            x1, len1, max_len=max_len
                        )  # (cur_len, bs)

                        gen_tokens = gen_tokens.cpu()
                        for i in range(bs):
                            seq = gen_tokens[:, i].tolist()
                            # find <eos> after optional <sep> handling (generate handles it)
                            eos_pos = None
                            for j in range(1, len(seq)):
                                if seq[j] == env.eos_index:
                                    eos_pos = j
                                    break
                            if eos_pos is None:
                                eos_pos = len(seq)
                            body = [t for t in seq[1:eos_pos] if t not in SPECIAL]
                            pred_strs[i] = tokens_to_str(body)

                    elif arch == "encoder_only":
                        # Some codepaths define an encoder.decode. If present, use it.
                        if hasattr(encoder, "decode"):
                            preds = encoder.decode(x1, len1, exp_len=max_len).cpu()  # (bs, <=max_len)
                            for i in range(bs):
                                seq = preds[i].tolist()
                                # stop at EOS
                                eos_pos = None
                                for j, t in enumerate(seq):
                                    if t == env.eos_index:
                                        eos_pos = j
                                        break
                                if eos_pos is None:
                                    eos_pos = len(seq)
                                body = [t for t in seq[:eos_pos] if t not in SPECIAL]
                                pred_strs[i] = tokens_to_str(body)
                        else:
                            # Fallback: project last hidden state through the head (rarely used in this repo)
                            enc_states = encoder(
                                "fwd",
                                x=x1,
                                lengths=len1,
                                causal=False,
                                src_enc=None,
                                src_len=None,
                                use_cache=False,
                            )  # (slen, bs, dim)
                            # Best-effort greedy on step 1
                            last = enc_states[-1]  # (bs, dim)
                            scores = encoder.proj(last)  # (bs, n_words)
                            next_words = torch.topk(scores, 1)[1].squeeze(1).cpu().tolist()
                            pred_strs = [env.id2word[w] if w not in SPECIAL else "" for w in next_words]

                    # ---------- Merge with metadata ----------
                    for i in range(bs):
                        inp = inp_strs[i]
                        pred = pred_strs[i]
                        try:
                            rows = meta_map.loc[inp]
                        except KeyError:
                            # No matching metadata row for this input; skip
                            continue
                        if isinstance(rows, pd.DataFrame):
                            recs = rows.to_dict("records")
                        else:
                            recs = [rows.to_dict()]
                        for rec in recs:
                            out = dict(rec)
                            out["prediction"] = pred
                            results.append(out)

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(output_dir, f"labels_{epoch}.csv"), index=False)
    return out_df
