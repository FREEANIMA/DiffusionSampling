import torch
import pandas as pd
import json
import random
from dataclasses import asdict, is_dataclass
from networks.tags_llm import Model
from huggingface_hub import hf_hub_download
from version import HUGGINGFACE

class TagGenration:
    def __init__(self, device="cuda", dtype=torch.bfloat16, console=None):
        self.device = device
        self.dtype = dtype
        self.MIN_TAG_FREQ = 15
        self.MIN_FIXED_TAGS = 15

        self.NAS = [
            "style",
            "person_count",
            "gender",
            "species",
            "hair_color",
            "hair_length",
            "hair_style",
            "eye_color",
            "body_type",
            "clothing",
            "accessory",
            "background",
            "pose",
            "expression",
            "composition",
            "gaze",
            "sex_act"
        ]

        self.INIT_PROMPT = "general img; "


        ckpt_path, vocab_path, tags_type_path = self._download()

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)

        chk = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        config = chk["config"]
        param = asdict(config) if is_dataclass(config) else dict(config)
        param["vocab_size"] = vocab_size 

        self.text_model = Model(**param)
        self.text_model.load_state_dict(chk["model"])
        self.text_model = self.text_model.to(device=device, dtype=dtype)
        self.text_model.eval()

        tags_type_df = pd.read_parquet(tags_type_path)

        self.type_map, self.dup_map = self.build_tag_maps(tags_type_df=tags_type_df)

        self.PAD_ID = self.vocab["<pad>"]
        self.CLS_ID = self.vocab["<cls>"]
        self.EOS_ID = self.vocab["<eos>"]
        self.UNK_ID = self.vocab["<unk>"]

        self.id_to_tag = {v: k for k, v in self.vocab.items()}
        self.console = console

    def _download(self):
        model_path = hf_hub_download(
            repo_id=f"{HUGGINGFACE}/tags_llm",
            filename="model.pt",
            repo_type="model",
        )
        vocab_path = hf_hub_download(
            repo_id=f"{HUGGINGFACE}/tags_llm",
            filename="vocab.json",
            repo_type="model",
        )
        tags_type_path = hf_hub_download(
            repo_id=f"{HUGGINGFACE}/tags_llm",
            filename="tags_type.parquet",
            repo_type="model",
        )

        return model_path, vocab_path, tags_type_path

    def build_tag_maps(self, tags_type_df):
        s = tags_type_df[["tag", "type", "dup", "count"]].drop_duplicates(subset="tag", keep="first")

        dup_map = dict(zip(s["tag"], s["dup"].fillna(s["tag"])))

        common = s[s["count"].fillna(0) >= self.MIN_TAG_FREQ]
        type_map = dict(zip(common["tag"], common["type"]))

        return type_map, dup_map

    def get_ac_tags(self):
        return sorted(t for t in self.type_map if t in self.vocab)

    def canonicalize(self, tag_list: list):
        out = []
        seen = set()
        for t in tag_list:
            t = t.strip()
            if not t:
                continue
            rep = self.dup_map.get(t, t)
            if rep not in seen:
                seen.add(rep)
                out.append(rep)
        return out

    def to_ids(self, xs, max_len=256):
        body = []
        for t in xs:
            body.append(self.vocab.get(t, self.UNK_ID))
            if len(body) >= max_len - 1:
                break
        return [self.CLS_ID] + body

    @torch.inference_mode()
    def sampling(self, batch, max_new=256, temperature=1.0, top_k=50, top_p=0.95, min_new=16):
        ids = batch
        generated = []

        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            logits, past = self.text_model(x=ids, attn_mask=None, use_cache=True)
            next_logits = logits[:, -1, :]            # (1, V)

            for step in range(max_new):
                next_logits = next_logits.float() / max(temperature, 1e-5)
                for bad in (self.PAD_ID, self.CLS_ID, self.UNK_ID):
                    next_logits[:, bad] = float("-inf")
                if step < min_new:                    # 최소 길이 동안 EOS 차단
                    next_logits[:, self.EOS_ID] = float("-inf")

                if top_k is not None and top_k > 0:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                tok = next_id.item()
                if tok == self.EOS_ID:
                    break
                generated.append(tok)

                logits, past = self.text_model(x=next_id, attn_mask=None, past_kv=past, use_cache=True)
                next_logits = logits[:, -1, :]

        return [self.id_to_tag.get(i, f"<{i}>") for i in generated]
    
    def text_generation(self, tags_infer, nsfw, H, W, user_tags=None):
        nsfw_text = f"nsfw: {str(nsfw).lower()}; "
        size_text = f"size: {H}x{W}; "
        merged, seen = [], set()

        for t in list(user_tags or []) + list(tags_infer):
            if t in ("<unk>", "<pad>", "<cls>", "<eos>"):
                continue
            rep = self.dup_map.get(t, t)
            if rep not in seen:
                seen.add(rep)
                merged.append(t)

        grouped, others = {}, []
        for tag in merged:
            _type = self.type_map.get(tag)
            if _type is None or pd.isna(_type) or not str(_type).strip():
                others.append(tag)
            else:
                grouped.setdefault(_type, []).append(tag)

        ordered_types = [t for t in self.NAS if t in grouped]
        ordered_types += [t for t in grouped if t not in self.NAS]

        text = "".join(f"{t}: {', '.join(grouped[t])}; " for t in ordered_types)
        others_text = f"others: {', '.join(others)}; " if others else ""
        return [self.INIT_PROMPT + nsfw_text + size_text + text + others_text]
    
    def tags_mapping_worker(self, tags: str, nsfw, H, W, device=None, fixed: bool = True, isnegative=False):
        device = device or self.device
        nsfw_str = "nsfw" if nsfw else "general"

        user_tags = self.canonicalize(tag_list=tags.split(","))

        def tags_s():
            known = [t for t in user_tags if t in self.vocab]
            dup_list = known + [nsfw_str]
            random.shuffle(dup_list)
            ids = self.to_ids(dup_list)
            batch = torch.tensor([ids], dtype=torch.long, device=device)
            tags_infer = self.sampling(batch)

            return tags_infer

        if isnegative:
            tags_infer = []
        elif fixed and len(user_tags) < self.MIN_FIXED_TAGS:
            self.console.put(f"tags 수 부족({len(user_tags)}/{self.MIN_FIXED_TAGS}): 안정을 위해 tag sampling")
            tags_infer = tags_s()
        elif fixed:
            tags_infer = []
        else:
            tags_infer = tags_s()
                
        return self.text_generation(tags_infer, nsfw, H, W, user_tags=user_tags), tags_infer

def test():
    tag_g = TagGenration()

    for _ in range(5):
        tag = tag_g.tags_mapping_worker(tags="woman", H="1024", W="1024", nsfw=True)
        print(tag)