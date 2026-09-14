"""B08 - truyen theo luong co lam mo hinh nhanh hon khong? Do BA dai luong tach bach:
(1) TTFT phia engine (doc /metrics cua vLLM), (2) luc client thay dau ra dau tien, (3) E2E."""
import asyncio, json, statistics, time, httpx
URL = "http://127.0.0.1:8011"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b08.json"
N, MAX_TOK = 20, 256
CAU = "Giải thích chi tiết vì sao bộ nhớ đệm khóa–giá trị giúp mô hình ngôn ngữ sinh văn bản nhanh hơn."

def doc_metrics(txt):
    d = {}
    for dong in txt.splitlines():
        if dong.startswith("vllm:") and ("_sum" in dong.split("{")[0] or "_count" in dong.split("{")[0]):
            ten = dong.split("{")[0].split(" ")[0]
            d[ten] = d.get(ten, 0.0) + float(dong.rsplit(" ", 1)[1])
    return d

def hieu(a, b, goc):
    s, c = b.get(goc + "_sum", 0) - a.get(goc + "_sum", 0), b.get(goc + "_count", 0) - a.get(goc + "_count", 0)
    return round(s / c, 4) if c else None

async def mot(cl, luong):
    body = dict(model=MODEL, messages=[{"role": "user", "content": CAU}], max_tokens=MAX_TOK,
                temperature=0, stream=luong, ignore_eos=True)
    t0 = time.perf_counter(); thay_dau = None
    if luong:
        async with cl.stream("POST", URL + "/v1/chat/completions", json=body) as r:
            async for dong in r.aiter_lines():
                if dong.startswith("data: ") and dong != "data: [DONE]":
                    j = json.loads(dong[6:])
                    if thay_dau is None and j["choices"] and j["choices"][0]["delta"].get("content"):
                        thay_dau = time.perf_counter() - t0
    else:
        r = await cl.post(URL + "/v1/chat/completions", json=body)
        r.raise_for_status(); thay_dau = time.perf_counter() - t0      # khong luong: thay chu dau tien = luc nhan het
    return thay_dau, time.perf_counter() - t0

async def main():
    res = dict(model=MODEL, n=N, max_tokens=MAX_TOK, luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    async with httpx.AsyncClient(timeout=600) as cl:
        await mot(cl, True); await mot(cl, False)                     # khoi dong
        ten_goc = None
        for luong in (True, False):
            m0 = doc_metrics((await cl.get(URL + "/metrics")).text)
            thay, e2e = [], []
            for _ in range(N):                                         # tuan tu: batch 1, khong tranh chap
                a, b = await mot(cl, luong); thay.append(a); e2e.append(b)
            m1 = doc_metrics((await cl.get(URL + "/metrics")).text)
            if ten_goc is None:
                res["ten_metrics_co_san"] = sorted({k.rsplit("_", 1)[0] for k in m1})
            che_do = "luong" if luong else "khong_luong"
            res[che_do] = dict(
                client_thay_dau_ra_dau_tien_s=round(statistics.median(thay), 3),
                client_e2e_s=round(statistics.median(e2e), 3),
                engine_ttft_s=hieu(m0, m1, "vllm:time_to_first_token_seconds"),
                engine_e2e_s=hieu(m0, m1, "vllm:e2e_request_latency_seconds"),
                engine_hang_doi_s=hieu(m0, m1, "vllm:request_queue_time_seconds"))
            print(che_do, res[che_do], flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("da ghi", OUT, flush=True)
asyncio.run(main())
