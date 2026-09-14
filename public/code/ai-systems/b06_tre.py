"""B06 phan 2 - DO TRE TUNG YEU CAU theo nhom do dai, tren tai lan lon.
Dung:  python b06_tre.py tinh | vllm16 | vllm256
Ghi thoi diem co token dau (TTFT) va thoi diem xong (E2E) cua tung yeu cau; tat ca den luc t=0."""
import json, statistics, sys, time
CHE_DO = sys.argv[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = f"/media/ai/SSD/vietpv/booksys/ket-qua/b06-tre-{CHE_DO}.json"
PROMPT, LO = 128, 16
DD = [32, 64, 128, 512] * 16
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
nen = tok(("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng "
           "trên một nhiệm vụ thông qua kinh nghiệm, dữ liệu và phản hồi. ") * 400).input_ids
ttft, e2e = {}, {}
if CHE_DO == "tinh":
    import torch
    from transformers import AutoModelForCausalLM
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    def xuoi(ids, cache=None):
        try: return mdl(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        except TypeError: return mdl(ids, past_key_values=cache, use_cache=True)
    @torch.no_grad()
    def chay(nhom, so_buoc, t0):
        ids = torch.tensor([nen[i * 37: i * 37 + PROMPT] for i in nhom], device="cuda")
        o = xuoi(ids); nxt = o.logits[:, -1:].argmax(-1); torch.cuda.synchronize()
        t_dau = time.perf_counter() - t0; cache = o.past_key_values
        for _ in range(so_buoc - 1):
            o = xuoi(nxt, cache); nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
        torch.cuda.synchronize(); return t_dau, time.perf_counter() - t0
    chay(range(LO), 8, time.perf_counter())
    t0 = time.perf_counter()
    for k in range(0, len(DD), LO):
        nhom = list(range(k, k + LO)); a, b = chay(nhom, max(DD[i] for i in nhom), t0)
        for i in nhom: ttft[i], e2e[i] = a, b
else:
    from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
    cho = 16 if CHE_DO == "vllm16" else 256
    eng = LLMEngine.from_engine_args(EngineArgs(model=MODEL, gpu_memory_utilization=0.8, max_model_len=2048,
                                                max_num_seqs=cho, seed=0))
    def them(tien_to, dd):
        for i, L in enumerate(dd):
            eng.add_request(f"{tien_to}{i}", TokensPrompt(prompt_token_ids=nen[i * 37: i * 37 + PROMPT]),
                            SamplingParams(temperature=0, max_tokens=L, ignore_eos=True))
    them("kd", [8] * 16)
    while eng.has_unfinished_requests(): eng.step()
    them("", DD); t0 = time.perf_counter()
    while eng.has_unfinished_requests():
        for o in eng.step():
            i = int(o.request_id); t = time.perf_counter() - t0
            if i not in ttft and len(o.outputs[0].token_ids) >= 1: ttft[i] = t
            if o.finished: e2e[i] = t
tong = max(e2e.values())
nhom = {}
for L in sorted(set(DD)):
    idx = [i for i, d in enumerate(DD) if d == L]
    nhom[L] = dict(ttft_trung_vi_s=round(statistics.median(ttft[i] for i in idx), 3),
                   e2e_trung_vi_s=round(statistics.median(e2e[i] for i in idx), 3))
res = dict(che_do=CHE_DO, model=MODEL, so_yeu_cau=len(DD), giay_ca_tai=round(tong, 2),
           thong_luong_tok_s=round(sum(DD) / tong, 1), theo_do_dai=nhom)
print(json.dumps(res, ensure_ascii=False), flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
