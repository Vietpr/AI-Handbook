"""B06 - moc doi chieu: gop lo TINH tu viet, lo 16 cho, tren hai tai: cung do dai va do dai lan lon.
Tong so token huu ich cua hai tai bang nhau, de so thang voi vLLM."""
import json, statistics, subprocess, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b06-tinh.json"
PROMPT, LO = 128, 16
TAI = {"cung_do_dai": [184] * 64, "lan_lon": [32, 64, 128, 512] * 16}   # ca hai: 11.776 token huu ich
sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
res = dict(model=MODEL, cho=LO, luc=time.strftime("%Y-%m-%d %H:%M:%S"),
           gpu=sh("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"))
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
nen = tok(("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng "
           "trên một nhiệm vụ thông qua kinh nghiệm, dữ liệu và phản hồi. ") * 400, return_tensors="pt").input_ids[0]
def xuoi(ids, cache=None):
    try: return mdl(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    except TypeError: return mdl(ids, past_key_values=cache, use_cache=True)
@torch.no_grad()
def chay(chi_so, so_buoc):
    ids = torch.stack([nen[i * 37: i * 37 + PROMPT] for i in chi_so]).to("cuda")
    o = xuoi(ids); nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
    for _ in range(so_buoc - 1):
        o = xuoi(nxt, cache); nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
    torch.cuda.synchronize()
chay(range(LO), 8)                                    # khoi dong
for ten, dd in TAI.items():
    torch.cuda.synchronize(); t0 = time.perf_counter(); xong = []
    for k in range(0, len(dd), LO):
        nhom = list(range(k, k + LO))
        chay(nhom, max(dd[i] for i in nhom))           # ca lo chay toi khi yeu cau dai nhat xong
        xong += [time.perf_counter() - t0] * LO        # lo tinh tra ket qua mot luot, luc ca lo xong
    tong = time.perf_counter() - t0
    huu_ich = sum(dd); o_chiem = sum(LO * max(dd[k:k + LO]) for k in range(0, len(dd), LO))
    xs = sorted(xong)
    res[ten] = dict(so_yeu_cau=len(dd), token_huu_ich=huu_ich, giay=round(tong, 2),
                    thong_luong_tok_s=round(huu_ich / tong, 1),
                    e2e_p50_s=round(xs[len(xs) // 2], 2), e2e_p95_s=round(xs[int(len(xs) * .95)], 2),
                    lang_phi_phan_tram=round(100 * (1 - huu_ich / o_chiem), 1))
    print(ten, res[ten], flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
