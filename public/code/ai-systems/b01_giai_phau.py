"""B01 - Giải phẫu một lần gọi mô hình: nạp prompt so với sinh token.
Đo TTFT, TPOT/ITL, E2E; và ước lượng băng thông bộ nhớ đạt được ở pha sinh."""
import json, os, statistics, subprocess, time, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT   = "/media/ai/SSD/vietpv/booksys/ket-qua/b01.json"
DO_DAI_PROMPT = [64, 256, 1024, 4096]
SO_TOKEN_SINH = 96
LAP = 5

def trang_thai_gpu():
    q = ("index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,"
         "temperature.gpu,clocks.current.sm,clocks.current.memory")
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(gpu=r, tien_trinh_dang_chay=p, luc=time.strftime("%Y-%m-%d %H:%M:%S"))

def main():
    bc = dict(model=MODEL, truoc_khi_chay=trang_thai_gpu())
    print("GPU trước khi chạy:", bc["truoc_khi_chay"]["gpu"], flush=True)
    if bc["truoc_khi_chay"]["tien_trinh_dang_chay"]:
        print("CẢNH BÁO - có tiến trình khác trên GPU:", bc["truoc_khi_chay"]["tien_trinh_dang_chay"], flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    n_tham_so = sum(p.numel() for p in mdl.parameters())
    byte_trong_so = sum(p.numel() * p.element_size() for p in mdl.parameters())
    cfg = mdl.config
    bc["model_info"] = dict(
        tham_so=n_tham_so, byte_trong_so=byte_trong_so,
        lop=cfg.num_hidden_layers, q_head=cfg.num_attention_heads,
        kv_head=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=str(next(mdl.parameters()).dtype))
    print("tham số:", n_tham_so, "| trọng số:", round(byte_trong_so/1e9, 2), "GB", flush=True)

    nen = ("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện "
           "hiệu năng trên một nhiệm vụ thông qua kinh nghiệm. ")
    ket = {}

    @torch.no_grad()
    def mot_luot(ids):
        """Trả về (ttft, [itl...], e2e) cho một lần gọi."""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = mdl(ids, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize(); ttft = time.perf_counter() - t0
        cache, itls = out.past_key_values, []
        for _ in range(SO_TOKEN_SINH - 1):
            t1 = time.perf_counter()
            o = mdl(nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            cache = o.past_key_values
            torch.cuda.synchronize(); itls.append(time.perf_counter() - t1)
        return ttft, itls, ttft + sum(itls)

    for L in DO_DAI_PROMPT:
        ids = tok(nen * (L // 20 + 2), return_tensors="pt").input_ids[:, :L].to("cuda")
        assert ids.shape[1] == L, ids.shape
        mot_luot(ids)  # khởi động
        ttfts, itls_all, e2es = [], [], []
        for _ in range(LAP):
            a, b, c = mot_luot(ids)
            ttfts.append(a); itls_all += b; e2es.append(c)
        itls_all.sort()
        r = dict(
            ttft_s=round(statistics.median(ttfts), 4),
            tpot_p50_ms=round(statistics.median(itls_all) * 1e3, 2),
            tpot_p95_ms=round(itls_all[int(len(itls_all) * .95)] * 1e3, 2),
            e2e_s=round(statistics.median(e2es), 3),
            nap_prompt_tok_s=round(L / statistics.median(ttfts)),
            sinh_tok_s=round(1 / statistics.median(itls_all), 1),
            vram_dinh_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        # băng thông đạt được ở pha sinh: mỗi token phải đọc toàn bộ trọng số một lần
        r["bang_thong_dat_GBps"] = round(byte_trong_so * r["sinh_tok_s"] / 1e9, 1)
        ket[L] = r
        print(L, r, flush=True)

    bc["gpu"] = ket
    bc["sau_khi_chay"] = trang_thai_gpu()
    json.dump(bc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("đã ghi", OUT)

main()
