"""B06 - cung hai tai ay qua vLLM. Dung:  python b06_vllm.py <max_num_seqs>
max_num_seqs=16 -> cung so cho voi lo tinh; mac dinh 256 -> xem tac dong cua viec tang so cho."""
import json, subprocess, sys, time
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHO = int(sys.argv[1])
OUT = f"/media/ai/SSD/vietpv/booksys/ket-qua/b06-vllm-{CHO}.json"
PROMPT = 128
TAI = {"cung_do_dai": [184] * 64, "lan_lon": [32, 64, 128, 512] * 16}
sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
res = dict(model=MODEL, max_num_seqs=CHO, luc=time.strftime("%Y-%m-%d %H:%M:%S"),
           gpu=sh("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"))
tok = AutoTokenizer.from_pretrained(MODEL)
nen = tok(("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng "
           "trên một nhiệm vụ thông qua kinh nghiệm, dữ liệu và phản hồi. ") * 400).input_ids
llm = LLM(model=MODEL, gpu_memory_utilization=0.8, max_model_len=2048, max_num_seqs=CHO, seed=0)
def prompts(n): return [TokensPrompt(prompt_token_ids=nen[i * 37: i * 37 + PROMPT]) for i in range(n)]
llm.generate(prompts(16), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
for ten, dd in TAI.items():
    sps = [SamplingParams(temperature=0, max_tokens=L, ignore_eos=True) for L in dd]
    t0 = time.perf_counter(); outs = llm.generate(prompts(len(dd)), sps, use_tqdm=False)
    tong = time.perf_counter() - t0
    dai = [len(o.outputs[0].token_ids) for o in outs]
    assert dai == dd, "do dai dau ra khac yeu cau"
    r = dict(so_yeu_cau=len(dd), token_huu_ich=sum(dd), giay=round(tong, 2),
             thong_luong_tok_s=round(sum(dd) / tong, 1))
    # thoi diem xong tung yeu cau, neu phien ban nay co ghi
    m = getattr(outs[0], "metrics", None)
    r["co_metrics"] = type(m).__name__ if m is not None else None
    if m is not None:
        r["truong_metrics"] = sorted(k for k in dir(m) if not k.startswith("_"))[:30]
    res[ten] = r; print(ten, r, flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
