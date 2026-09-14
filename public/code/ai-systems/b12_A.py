"""B12 chang A - moc goc: dung y he o chuong Generative AI - Qwen2.5-0.5B, transformers, TUAN TU tung cau."""
import json, os, statistics, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b12_chung as C
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; MAXT = 200
d = C.nap(); muc = d["muc"]
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
torch.cuda.reset_peak_memory_stats()
@torch.no_grad()
def goi(m):
    txt = tok.apply_chat_template(C.tin_nhan(m), tokenize=False, add_generation_prompt=True)
    ids = tok(txt, return_tensors="pt").input_ids.to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o = mdl(ids, use_cache=True, logits_to_keep=1); nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0; ra = [nxt.item()]
    for _ in range(MAXT - 1):
        if ra[-1] == tok.eos_token_id or ra[-1] in (151645, 151643): break
        o = mdl(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1); nxt = o.logits[:, -1:].argmax(-1)
        cache = o.past_key_values; ra.append(nxt.item())
    torch.cuda.synchronize(); return ttft, time.perf_counter() - t0, tok.decode(ra, skip_special_tokens=True), ids.shape[1], len(ra)
goi(muc[0])
t0 = time.perf_counter(); kq = [goi(m) for m in muc]; tong = time.perf_counter() - t0
ttft = sorted(k[0] for k in kq); e2e = sorted(k[1] for k in kq)
res = dict(chang="A", mo_ta="0.5B, transformers, tuan tu", so_yeu_cau=len(kq), giay=round(tong, 2),
           req_s=round(len(kq) / tong, 3), tok_ra_s=round(sum(k[4] for k in kq) / tong, 1),
           ttft_p50=round(ttft[len(ttft) // 2], 3), ttft_p95=round(ttft[int(len(ttft) * .95)], 3),
           e2e_p50=round(e2e[len(e2e) // 2], 3), e2e_p95=round(e2e[int(len(e2e) * .95)], 3),
           token_vao_tb=round(statistics.mean(k[3] for k in kq)), vram_dinh_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
           chat_luong=C.cham(muc, [k[2] for k in kq]), dau_ra=[k[2] for k in kq])
json.dump(res, open("/media/ai/SSD/vietpv/booksys/ket-qua/b12-A.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in res.items() if k != "dau_ra"}, flush=True)
