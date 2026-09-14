"""B03 phan 3 - nap ban 7B len card 12 GB de bat loi OOM that."""
import json, os, glob, torch, subprocess
from transformers import AutoModelForCausalLM, AutoConfig
TO="Qwen/Qwen2.5-Coder-7B-Instruct"
OUT="/media/ai/SSD/vietpv/booksys/ket-qua/b03-oom.json"
tong=torch.cuda.get_device_properties(0).total_memory
idx=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/model.safetensors.index.json"))
ts=json.load(open(idx[0]))["metadata"]["total_size"] if idx else None
r=dict(checkpoint_byte=ts, checkpoint_GB=round(ts/1e9,2) if ts else None,
       vram_card_GB=round(tong/1e9,2), thua_GB=round((ts-tong)/1e9,2) if ts else None)
print(r, flush=True)
print("=== thu nap 7B bf16 len 1 card ===", flush=True)
try:
    m=AutoModelForCausalLM.from_pretrained(TO, dtype=torch.bfloat16).to("cuda")
    r["ket_qua"]="nap duoc - PHAI KIEM LAI GIA DINH"
    print(r["ket_qua"], flush=True)
except Exception as e:
    t=str(e)
    r["ket_qua"]="OOM"; r["loai_loi"]=type(e).__name__; r["thong_bao"]=t[:1500]
    print(type(e).__name__); print(t[:1000], flush=True)
r["smi_sau"]=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader"],
                            capture_output=True,text=True).stdout.strip()
json.dump(r, open(OUT,"w"), indent=1, ensure_ascii=False)
print("da ghi", OUT)
