"""B05 - Gop lo tinh: thong luong, do tre, bo nho theo batch; va cai gia cua 'thang cham nhat'.
Cung mo hinh va cung card voi bai 01 de so thang duoc."""
import json, statistics, subprocess, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b05.json"
TRAN_TINH, TRAN_BW = 27.63e12, 318.3e9       # tran do o bai 01
BATCH = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT, SINH = 128, 128

sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
res = dict(model=MODEL, luc=time.strftime("%Y-%m-%d %H:%M:%S"),
           gpu=sh("nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"),
           tien_trinh_khac=sh("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"))
print("GPU:", res["gpu"], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
cfg = mdl.config
P = sum(p.numel() for p in mdl.parameters())
WB = sum(p.numel() * p.element_size() for p in mdl.parameters())
KV = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2

nen = tok(("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng "
           "trên một nhiệm vụ thông qua kinh nghiệm, dữ liệu và phản hồi. ") * 400,
          return_tensors="pt").input_ids[0]

def lo(B):
    """B prompt KHAC NHAU nhung CUNG do dai PROMPT -> khong can dem."""
    return torch.stack([nen[i * 37: i * 37 + PROMPT] for i in range(B)]).to("cuda")

def xuoi(ids, cache=None):
    try:
        return mdl(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    except TypeError:
        return mdl(ids, past_key_values=cache, use_cache=True)

@torch.no_grad()
def chay_lo(ids, so_buoc):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o = xuoi(ids); nxt = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
    cache, buoc = o.past_key_values, []
    for _ in range(so_buoc - 1):
        t1 = time.perf_counter()
        o = xuoi(nxt, cache); nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
        torch.cuda.synchronize(); buoc.append(time.perf_counter() - t1)
    return ttft, buoc

# ---------- 1. quet batch ----------
quet = {}
for B in BATCH:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = lo(B); chay_lo(ids, 4)
        ttft, buoc = chay_lo(ids, SINH)
        tpot = statistics.median(buoc)
        ctx = PROMPT + SINH / 2
        flop_s = 2 * P * B / tpot
        byte_s = (WB + B * ctx * KV) / tpot
        r = dict(ttft_s=round(ttft, 4), tpot_ms=round(tpot * 1e3, 2),
                 tok_s_moi_yeu_cau=round(1 / tpot, 1), thong_luong_tok_s=round(B / tpot, 1),
                 e2e_s=round(ttft + sum(buoc), 3), vram_dinh_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 phan_tram_tran_tinh=round(100 * flop_s / TRAN_TINH, 2),
                 phan_tram_tran_bw=round(100 * byte_s / TRAN_BW, 1),
                 mat_do_FLOP_byte=round(2 * P * B / (WB + B * ctx * KV), 1))
        quet[B] = r; print(B, r, flush=True)
    except torch.OutOfMemoryError as e:
        quet[B] = dict(oom=str(e).splitlines()[0][:200]); print(B, "OOM", flush=True); break
res["quet_batch"] = quet

# ---------- 2. thang cham nhat: 8 yeu cau ngan + 8 yeu cau dai trong cung mot lo tinh ----------
ngan, dai = 32, 256
ids = lo(16); torch.cuda.empty_cache()
ttft, buoc = chay_lo(ids, dai)                     # lo tinh: ca lo chay toi khi yeu cau dai nhat xong
t_lo = ttft + sum(buoc)
t_ngan_rieng = sum(chay_lo(lo(8), ngan)[1]) + chay_lo(lo(8), ngan)[0]   # 8 yeu cau ngan chay rieng
huu_ich = 8 * ngan + 8 * dai
o_chiem = 16 * dai
res["thang_cham_nhat"] = dict(
    so_yeu_cau=16, ngan_token=ngan, dai_token=dai,
    token_huu_ich=huu_ich, o_tinh_toan_da_dung=o_chiem,
    lang_phi_phan_tram=round(100 * (1 - huu_ich / o_chiem), 1),
    e2e_yeu_cau_ngan_trong_lo_tinh_s=round(t_lo, 3),
    e2e_yeu_cau_ngan_neu_chay_rieng_s=round(t_ngan_rieng, 3),
    cham_hon_lan=round(t_lo / t_ngan_rieng, 1))
print("thang cham nhat:", res["thang_cham_nhat"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
