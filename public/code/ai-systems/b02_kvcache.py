"""B02 - KV cache: cong thuc, do truc tiep, do doc bo nho, va cai gia cua viec bo cache."""
import json, statistics, subprocess, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT   = "/media/ai/SSD/vietpv/booksys/ket-qua/b02.json"

def gpu_state():
    q = "index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu"
    a = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    b = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(gpu=a, tien_trinh_khac=b, luc=time.strftime("%Y-%m-%d %H:%M:%S"))

def cong_thuc(cfg, byte_moi_so=2):
    """KV byte cho MOT token: 2 (K va V) x so_lop x so_dau_KV x head_dim x byte."""
    hd = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * byte_moi_so

def lay_tensor_kv(pkv):
    """Tra ve danh sach tensor K,V bat ke kieu cache cua phien ban transformers nao."""
    ts = []
    if hasattr(pkv, "layers"):
        for lay in pkv.layers:
            for ten in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(lay, ten, None)
                if torch.is_tensor(t): ts.append(t)
    if not ts and hasattr(pkv, "key_cache"):
        ts = [t for t in list(pkv.key_cache) + list(pkv.value_cache) if torch.is_tensor(t)]
    if not ts:
        try:
            for k, v in pkv: ts += [k, v]
        except Exception: pass
    return ts

res = dict(model=MODEL, truoc=gpu_state())
print("GPU:", res["truoc"]["gpu"], flush=True)
print("tien trinh khac:", res["truoc"]["tien_trinh_khac"] or "(khong co)", flush=True)

# ---------- 1. CONG THUC cho ba mo hinh, khong can nap ----------
bang_ct = {}
for m in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
    c = AutoConfig.from_pretrained(m)
    hd = c.hidden_size // c.num_attention_heads
    b_gqa = cong_thuc(c)
    b_mha = 2 * c.num_hidden_layers * c.num_attention_heads * hd * 2
    bang_ct[m] = dict(lop=c.num_hidden_layers, q=c.num_attention_heads, kv=c.num_key_value_heads,
                      head_dim=hd, byte_moi_token=b_gqa, KiB_moi_token=round(b_gqa/1024, 1),
                      neu_MHA_KiB=round(b_mha/1024, 1), ty_le_GQA=c.num_attention_heads//c.num_key_value_heads)
    print(m, bang_ct[m], flush=True)
res["cong_thuc"] = bang_ct

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
cfg = mdl.config
ct = cong_thuc(cfg)
nen = "Hoc may la nganh nghien cuu cac thuat toan cho phep may tinh cai thien hieu nang. "

# ---------- 2. DO TRUC TIEP kich thuoc tensor cache ----------
do_tt = {}
for T in (512, 2048):
    ids = tok(nen * (T // 14 + 3), return_tensors="pt").input_ids[:, :T].to("cuda")
    with torch.no_grad():
        o = mdl(ids, use_cache=True)
    ts = lay_tensor_kv(o.past_key_values)
    tong = sum(t.numel() * t.element_size() for t in ts)
    do_tt[T] = dict(so_tensor=len(ts), tong_byte=tong, byte_moi_token=round(tong / T, 1),
                    dang_mot_tensor=str(tuple(ts[0].shape)) if ts else None,
                    kieu=str(ts[0].dtype) if ts else None)
    print("do truc tiep T=", T, do_tt[T], flush=True)
    del o; torch.cuda.empty_cache()
res["do_truc_tiep"] = do_tt

# ---------- 3. DO DOC BO NHO GPU theo do dai ngu canh ----------
doc = {}
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
nen_tang = torch.cuda.memory_allocated()
for T in (512, 1024, 2048, 4096, 8192):
    ids = tok(nen * (T // 14 + 3), return_tensors="pt").input_ids[:, :T].to("cuda")
    torch.cuda.empty_cache(); truoc = torch.cuda.memory_allocated()
    with torch.no_grad():
        o = mdl(ids, use_cache=True)
    ts = lay_tensor_kv(o.past_key_values)
    kv_byte = sum(t.numel() * t.element_size() for t in ts)
    sau = torch.cuda.memory_allocated()
    doc[T] = dict(kv_byte=kv_byte, delta_cap_phat=sau - truoc, dinh_GB=round(torch.cuda.max_memory_allocated()/1e9, 3))
    print("doc T=", T, doc[T], flush=True)
    del o; torch.cuda.empty_cache()
xs = sorted(doc); ys = [doc[t]["kv_byte"] for t in xs]
do_doc = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
res["doc_bo_nho"] = dict(diem=doc, do_doc_byte_moi_token=round(do_doc, 1),
                         cong_thuc_byte_moi_token=ct, sai_lech_phan_tram=round(100*(do_doc-ct)/ct, 3))
print("DO DOC:", do_doc, "| CONG THUC:", ct, flush=True)

# ---------- 4. CAI GIA CUA VIEC BO CACHE ----------
gia = {}
ids = tok(nen * 8, return_tensors="pt").input_ids[:, :128].to("cuda")
for N in (32, 64, 128, 256):
    r = {}
    for dung_cache in (True, False):
        with torch.no_grad():
            mdl.generate(ids, max_new_tokens=4, do_sample=False, use_cache=dung_cache)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            mdl.generate(ids, max_new_tokens=N, do_sample=False, use_cache=dung_cache)
            torch.cuda.synchronize(); r["co" if dung_cache else "khong"] = round(time.perf_counter()-t0, 3)
    r["ty_le"] = round(r["khong"] / r["co"], 2)
    gia[N] = r; print("bo cache N=", N, r, flush=True)
res["gia_bo_cache"] = gia

res["sau"] = gpu_state()
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT)
