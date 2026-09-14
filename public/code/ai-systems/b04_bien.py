"""B04 - bien bo nho cua ban 7B AWQ SAU KHI da vua mot card. vLLM khong sap khi vuot bien:
no day yeu cau ra roi tinh lai (preemption). Nen do: so lan preemption, thoi gian, thong luong,
doi chieu voi du doan tu vung KV engine tu bao."""
import json, time
from vllm import LLM, SamplingParams, TokensPrompt
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b04-bien.json"
GMU = 0.8   # lan dau o 0.9 thi OOM ngay diem thu hai - xem bai 04
llm = LLM(model=MODEL, quantization="awq_marlin", max_model_len=4096, gpu_memory_utilization=GMU,
          max_num_seqs=256, seed=0, disable_log_stats=False)
tok = llm.get_tokenizer()
nen = tok(("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng "
           "trên một nhiệm vụ thông qua kinh nghiệm, dữ liệu và phản hồi. ") * 900).input_ids
def dem(ten):
    try:
        return sum(getattr(m, "value", 0) or 0 for m in llm.get_metrics() if m.name == ten)
    except Exception:
        return None
res = dict(model=MODEL, gpu_memory_utilization=GMU, luc=time.strftime("%Y-%m-%d %H:%M:%S"), ten_metrics=None, quet={})
try: res["ten_metrics"] = sorted({m.name for m in llm.get_metrics()})
except Exception as e: res["ten_metrics"] = f"khong doc duoc: {type(e).__name__}"
SINH = 128
for ctx in (2048, 4096):
    for C in (4, 8, 16, 24, 32, 48):
        L = ctx - SINH
        ps = [TokensPrompt(prompt_token_ids=nen[i * 13: i * 13 + L]) for i in range(C)]
        p0 = dem("vllm:num_preemptions")
        t = time.perf_counter()
        outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=SINH, ignore_eos=True), use_tqdm=False)
        dt = time.perf_counter() - t
        p1 = dem("vllm:num_preemptions")
        r = dict(ngu_canh=ctx, dong_thoi=C, token_trong_KV_neu_chay_het=C * ctx,
                 giay=round(dt, 2), token_ra_moi_giay=round(C * SINH / dt, 1),
                 preemption=(p1 - p0) if (p0 is not None and p1 is not None) else None,
                 du_xong=all(len(o.outputs[0].token_ids) == SINH for o in outs))
        res["quet"][f"{ctx}x{C}"] = r; print(r, flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
