"""B03 - Giai phau bo nho GPU: ba cach dem khac nhau, va mot lan OOM that."""
import json, os, subprocess, time, glob, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

NHO = "Qwen/Qwen2.5-1.5B-Instruct"
TO  = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b03.json"

def smi_mib():
    """Bo nho ma driver bao tien trinh nay dang chiem tren GPU."""
    pid = os.getpid()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for d in out.strip().splitlines():
        p, m = [x.strip() for x in d.split(",")]
        if int(p) == pid: return int(m)
    return 0

def ba_so(nhan):
    a = torch.cuda.memory_allocated()/2**20
    r = torch.cuda.memory_reserved()/2**20
    s = smi_mib()
    d = dict(moc=nhan, cap_phat_MiB=round(a,1), giu_cho_MiB=round(r,1), driver_bao_MiB=s,
             chenh_giu_cho_tru_cap_phat=round(r-a,1), chenh_driver_tru_giu_cho=round(s-r,1))
    print(d, flush=True); return d

res = dict(luc=time.strftime("%Y-%m-%d %H:%M:%S"))
res["gpu"] = subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used",
                             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("GPU:", res["gpu"], flush=True)
tong_vram = torch.cuda.get_device_properties(0).total_memory
res["vram_tong_byte"] = tong_vram

# ---------- 1. SO HOC cho ban 7B, doc tu chinh checkpoint ----------
cfg7 = AutoConfig.from_pretrained(TO)
idx = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/model.safetensors.index.json"))
tong_ts = None
if idx:
    j = json.load(open(idx[0]))
    tong_ts = j.get("metadata", {}).get("total_size")
res["so_hoc_7B"] = dict(
    lop=cfg7.num_hidden_layers, q=cfg7.num_attention_heads, kv=cfg7.num_key_value_heads,
    vocab=cfg7.vocab_size, dtype=str(getattr(cfg7,"dtype",None) or getattr(cfg7,"torch_dtype",None)),
    tong_size_tu_checkpoint_byte=tong_ts,
    tong_size_GB=round(tong_ts/1e9,2) if tong_ts else None,
    vram_card_GB=round(tong_vram/1e9,2),
    thua_ra_GB=round((tong_ts-tong_vram)/1e9,2) if tong_ts else None)
print("7B:", res["so_hoc_7B"], flush=True)

# ---------- 2. BA CACH DEM tren ban 1.5B ----------
moc = []
torch.cuda.init(); torch.cuda.synchronize()
moc.append(ba_so("sau khi khoi tao CUDA, chua cap phat gi"))
ctx = smi_mib()
res["cuda_context_MiB"] = ctx

tok = AutoTokenizer.from_pretrained(NHO)
mdl = AutoModelForCausalLM.from_pretrained(NHO, dtype=torch.bfloat16).to("cuda").eval()
moc.append(ba_so("sau khi nap trong so 1.5B"))
byte_ts = sum(p.numel()*p.element_size() for p in mdl.parameters())
res["trong_so_1_5B_MiB"] = round(byte_ts/2**20,1)

nen = "Hoc may la nganh nghien cuu cac thuat toan cho phep may tinh cai thien hieu nang. "
ids = tok(nen*600, return_tensors="pt").input_ids[:, :4096].to("cuda")
with torch.no_grad(): o = mdl(ids, use_cache=True)
moc.append(ba_so("sau mot luot chay xuoi 4096 token"))
lg = o.logits.numel()*o.logits.element_size()
kv = sum(t.numel()*t.element_size() for lay in o.past_key_values.layers
         for t in (lay.keys, lay.values) if torch.is_tensor(t))
res["luot_xuoi_4096"] = dict(logits_MiB=round(lg/2**20,1), kv_MiB=round(kv/2**20,1))
del o; torch.cuda.empty_cache()
moc.append(ba_so("sau khi giai phong va don dep"))
res["cac_moc"] = moc

# (phan 3 tach sang b03_oom.py de khong phai tai lai 15 GB)

json.dump(res, open(OUT,"w"), indent=1, ensure_ascii=False)
print("\nda ghi", OUT)
