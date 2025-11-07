import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalDeepSeek:
    """
    Local DeepSeek runner using HF transformers (4-bit quantization).
    It returns a Python dict with:
      {"alpha": [...], "edges": [[i,j,score], ...],
       "film": {"img_gamma": [...], "img_beta": [...], "obj_gamma": [...], "obj_beta": [...]}}
    """
    def __init__(self, model_id="deepseek-ai/DeepSeek-V2-Lite", device_map="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
            load_in_4bit=True,             # needs bitsandbytes
            bnb_4bit_compute_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

        # target sizes for FiLM vectors
        self.img_channels = 64   # must match your CNN channels
        self.obj_dim      = 128  # must match your d_model

    def _prompt(self, obj_list):
        # obj_list: [(x,y,w,h,tid), ...] as ints/floats
        rows = []
        for i, (x,y,w,h,tid) in enumerate(obj_list):
            rows.append(f"[{i}] tid={int(tid)} xywh=({int(x)},{int(y)},{int(w)},{int(h)})")
        obj_txt = "\n".join(rows) if rows else "(none)"
        return (
            "Task: Atari Freeway. Agent=player(tid=1). Cars tid=2. Goal tid=3.\n"
            "Return STRICT JSON with:\n"
            '{"alpha":[...], "edges":[[i,j,score], ...], '
            '"film":{"img_gamma":[%d],"img_beta":[%d],"obj_gamma":[%d],"obj_beta":[%d]}}\n' 
            "alpha in [-1,1] per slot, edges score in [-1,1].\n"
            "Objects:\n%s\n"
            "JSON:" % (self.img_channels, self.img_channels, self.obj_dim, self.obj_dim, obj_txt)
        )

    @torch.inference_mode()
    def guidance(self, obj_list):
        prompt = self._prompt(obj_list)
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inp,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # extract JSON payload
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            js = json.loads(m.group(0))
            return js
        except Exception:
            return None
