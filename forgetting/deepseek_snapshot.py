from huggingface_hub import snapshot_download
snapshot_download("deepseek-ai/DeepSeek-V2-Lite", local_dir="models/deepseek-v2-lite")
# test_local_guidance.py
from local_deepseek import LocalDeepSeek
llm = LocalDeepSeek("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")  # or V2-Lite
objs = [(72,150,8,8,1), (70,150,8,8,2), (20,30,8,8,3)]  # [player, car, goal]
js = llm.guidance(objs)
print(js)  # expect dict with alpha/edges/film
