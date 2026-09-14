"""B09 phan A - bo dem tien to (prefix caching) cua vLLM. Dung:  python b09_tiento.py bat|tat
Do do tre NAP PROMPT cua tung yeu cau (max_tokens=1, chay tuan tu) khi nhieu yeu cau dung chung
mot system prompt dai; roi kiem: giai ma tat dinh co cho dau ra giong het khong, va lay mau thi sao."""
import json, sys, time
from vllm import LLM, SamplingParams
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
BAT = sys.argv[1] == "bat"
OUT = f"/media/ai/SSD/vietpv/booksys/ket-qua/b09-tiento-{sys.argv[1]}.json"
llm = LLM(model=MODEL, quantization="awq_marlin", max_model_len=4096, gpu_memory_utilization=0.85,
          enable_prefix_caching=BAT, seed=0, disable_log_stats=False)
tok = llm.get_tokenizer()
dieu = ("Điều {i}. Nhân viên phải tuân thủ quy định về an toàn thông tin, bảo mật dữ liệu khách hàng, "
        "sử dụng tài sản công ty đúng mục đích, báo cáo sự cố trong vòng 24 giờ, và tham gia đào tạo định kỳ. "
        "Mọi ngoại lệ phải được trưởng bộ phận phê duyệt bằng văn bản. ")
he_thong = "Bạn là trợ lý nội bộ. Chỉ trả lời dựa trên quy chế sau.\n" + "".join(dieu.format(i=i) for i in range(1, 41))
hoi = [f"Theo quy chế, điều {i} yêu cầu báo cáo sự cố trong bao lâu?" for i in range(1, 25)]
def p(q): return tok.apply_chat_template([{"role": "system", "content": he_thong}, {"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
res = dict(model=MODEL, prefix_caching=BAT, token_system_prompt=len(tok(he_thong).input_ids),
           token_moi_prompt=len(tok(p(hoi[0])).input_ids), luc=time.strftime("%Y-%m-%d %H:%M:%S"))
print("system prompt:", res["token_system_prompt"], "token | ca prompt:", res["token_moi_prompt"], flush=True)
khoi = tok.apply_chat_template([{"role": "user", "content": "Xin chào"}], tokenize=False, add_generation_prompt=True)
llm.generate([khoi], SamplingParams(max_tokens=1), use_tqdm=False)   # khoi dong nhan - that su KHONG kem system prompt
nap = []
for q in hoi:
    t = time.perf_counter(); llm.generate([p(q)], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    nap.append(round(time.perf_counter() - t, 4))
res["nap_prompt_tung_yeu_cau_s"] = nap
res["yeu_cau_dau_s"] = nap[0]
res["cac_yeu_cau_sau_trung_vi_s"] = sorted(nap[1:])[len(nap[1:]) // 2]
print("yeu cau dau:", nap[0], "| cac yeu cau sau (trung vi):", res["cac_yeu_cau_sau_trung_vi_s"], flush=True)
# chay TUAN TU tung prompt: tranh nap 8 x 2.843 token cung luc (lan truoc OOM khi tat bo dem)
res["dau_ra_tat_dinh"] = [llm.generate([p(q)], SamplingParams(max_tokens=64, temperature=0), use_tqdm=False)[0].outputs[0].text
                          for q in hoi[:8]]
lm = [llm.generate([p(hoi[0])], SamplingParams(max_tokens=48, temperature=0.8), use_tqdm=False)[0].outputs[0].text
      for _ in range(3)]
res["lay_mau_3_lan_cung_prompt"] = lm
res["lay_mau_khac_nhau"] = len(set(lm))
try:
    res["metrics_prefix"] = {m.name: getattr(m, "value", None) for m in llm.get_metrics() if "prefix" in m.name}
except Exception as e:
    res["metrics_prefix"] = f"khong doc duoc: {type(e).__name__}"
print("lay mau: so ban khac nhau trong 3 lan =", res["lay_mau_khac_nhau"], "| metrics:", res["metrics_prefix"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
