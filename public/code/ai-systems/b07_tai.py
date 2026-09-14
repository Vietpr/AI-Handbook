"""B07 - do tai phuc vu cho dung: tai DONG (C nguoi dung, gui tiep ngay khi xong) va
tai MO (yeu cau den theo Poisson o toc do lambda co dinh). Lay mau so yeu cau dang do moi 0,1 s
de kiem L = lambda * W, va xem hang doi co lon dan khong (he khong on dinh)."""
import asyncio, json, random, statistics, time, httpx
URL = "http://127.0.0.1:8011/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b07.json"
MAX_TOK = 128
CAU = "Viết một đoạn văn ngắn về lịch sử của máy tính cá nhân."
def pv(xs, q): xs = sorted(xs); return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3)

async def mot(cl, dang_do):
    body = dict(model=MODEL, messages=[{"role": "user", "content": CAU}], max_tokens=MAX_TOK,
                temperature=0, stream=True, ignore_eos=True)
    dang_do[0] += 1; t0 = time.perf_counter(); ttft = None
    try:
        async with cl.stream("POST", URL, json=body) as r:
            async for dong in r.aiter_lines():
                if ttft is None and dong.startswith("data: ") and '"content":"' in dong:
                    ttft = time.perf_counter() - t0
    finally:
        dang_do[0] -= 1
    return ttft, time.perf_counter() - t0

async def lay_mau(dang_do, mau, dung):
    while not dung.is_set():
        mau.append((time.perf_counter(), dang_do[0])); await asyncio.sleep(0.1)

def tong_ket(kq, mau, t_bd, t_kt, lam=None):
    ttft = [a for a, _ in kq if a is not None]; e2e = [b for _, b in kq]
    dur = t_kt - t_bd; lam_do = len(kq) / dur
    L = statistics.mean(v for _, v in mau) if mau else 0
    W = statistics.mean(e2e)
    nua = len(mau) // 2
    dau = statistics.mean(v for _, v in mau[:nua]) if nua else 0
    cuoi = statistics.mean(v for _, v in mau[nua:]) if nua else 0
    return dict(so_yeu_cau=len(kq), giay=round(dur, 1), lambda_dat=round(lam_do, 3),
                lambda_dat_ra=lam, thong_luong_tok_s=round(len(kq) * MAX_TOK / dur, 1),
                ttft_p50=pv(ttft, .5), ttft_p95=pv(ttft, .95), ttft_p99=pv(ttft, .99),
                e2e_p50=pv(e2e, .5), e2e_p95=pv(e2e, .95), e2e_p99=pv(e2e, .99), e2e_tb=round(W, 3),
                L_do=round(L, 2), lambda_nhan_W=round(lam_do * W, 2),
                dang_do_nua_dau=round(dau, 2), dang_do_nua_sau=round(cuoi, 2))

async def tai_dong(cl, C, so_moi_nguoi=6):
    dang_do, mau, dung, kq = [0], [], asyncio.Event(), []
    async def nguoi():
        for _ in range(so_moi_nguoi): kq.append(await mot(cl, dang_do))
    tm = asyncio.create_task(lay_mau(dang_do, mau, dung)); t0 = time.perf_counter()
    await asyncio.gather(*[nguoi() for _ in range(C)])
    t1 = time.perf_counter(); dung.set(); await tm
    return tong_ket(kq, mau, t0, t1)

async def tai_mo(cl, lam, giay=60):
    random.seed(int(lam * 100))
    dang_do, mau, dung, viec = [0], [], asyncio.Event(), []
    tm = asyncio.create_task(lay_mau(dang_do, mau, dung)); t0 = time.perf_counter()
    while time.perf_counter() - t0 < giay:
        viec.append(asyncio.create_task(mot(cl, dang_do)))
        await asyncio.sleep(random.expovariate(lam))
    t_het_den = time.perf_counter()
    kq = await asyncio.gather(*viec)
    t1 = time.perf_counter(); dung.set(); await tm
    r = tong_ket(kq, [m for m in mau if m[0] <= t_het_den], t0, t1, lam)
    r["giay_xa_hang_sau_khi_ngung_den"] = round(t1 - t_het_den, 1)
    return r

async def main():
    res = dict(model=MODEL, max_tokens=MAX_TOK, luc=time.strftime("%Y-%m-%d %H:%M:%S"), dong={}, mo={})
    lim = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    async with httpx.AsyncClient(timeout=900, limits=lim) as cl:
        await mot(cl, [0])
        for C in (1, 2, 4, 8, 16, 32, 64):
            res["dong"][C] = await tai_dong(cl, C); print("dong", C, res["dong"][C], flush=True)
        for lam in (0.5, 1, 2, 4, 8, 12):
            res["mo"][lam] = await tai_mo(cl, lam); print("mo", lam, res["mo"][lam], flush=True)
            json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("da ghi", OUT, flush=True)
asyncio.run(main())
