"""B12 - phan dung chung: cau hoi, prompt, cham chat luong. Moi chang deu import tep nay."""
import json, re, unicodedata
NGU_CANH = "/media/ai/SSD/vietpv/booksys/ket-qua/b12-ngucanh.json"
HE_THONG = (
 "Bạn là trợ lý hỏi đáp về một loạt bài học trí tuệ nhân tạo. Quy tắc:\n"
 "1. Chỉ trả lời dựa trên các đoạn trích được đánh số trong câu hỏi.\n"
 "2. Mỗi ý phải kèm số nguồn dạng [1], [2] hoặc [3].\n"
 "3. Nếu các đoạn trích không chứa câu trả lời, trả lời đúng một câu: \"Không tìm thấy trong tài liệu.\"\n"
 "4. Trả lời ngắn gọn, bằng tiếng Việt.\n\n"
 "Ví dụ 1.\nĐoạn trích:\n[1] Overfitting xảy ra khi mô hình học thuộc cả nhiễu của tập huấn luyện, nên sai số trên tập kiểm tra cao hơn hẳn tập huấn luyện.\n"
 "[2] Chia dữ liệu thành train, validation và test giúp phát hiện overfitting sớm.\n[3] Học máy cần dữ liệu để học.\n"
 "Câu hỏi: Dấu hiệu của overfitting là gì?\n"
 "Trả lời: Sai số trên tập kiểm tra cao hơn hẳn tập huấn luyện, vì mô hình đã học thuộc cả nhiễu [1]. Theo dõi tập validation giúp phát hiện sớm [2].\n\n"
 "Ví dụ 2.\nĐoạn trích:\n[1] Mạng tích chập dùng chung trọng số trên mọi vị trí ảnh.\n[2] Pooling giảm kích thước không gian.\n[3] ReLU là hàm kích hoạt phổ biến.\n"
 "Câu hỏi: Giá cổ phiếu hôm nay là bao nhiêu?\nTrả lời: Không tìm thấy trong tài liệu.\n")
# Cau ca koi KHONG co dap an trong kho - nhan "ngoai kho" van DUNG - nhung chinh cau hoi va mot cau tra loi sai
# da RO VAO KHO (chuong Generative AI - Bai 12 dung no lam vi du). Nen no la mot CAU AM KHO bi ro ri, khong phai
# cau am sach. Bo cham tinh "tu choi dung" tren cac cau am SACH, va bao rieng cau am kho.
AM_KHO_RO_RI = {"Chương này nói gì về cách nuôi cá koi trong hồ xi măng?"}
NGOAI_KHO = ["Chương này nói gì về cách nuôi cá koi trong hồ xi măng?", "Công thức nấu phở bò chuẩn vị Hà Nội là gì?",
             "Giá vàng hôm nay là bao nhiêu?", "Đội nào vô địch World Cup 2018?"]
def tin_nhan(muc):
    doan = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(muc["doan"]))
    return [{"role": "system", "content": HE_THONG},
            {"role": "user", "content": f"Đoạn trích:\n{doan}\nCâu hỏi: {muc['hoi']}"}]
def _chuan(x):
    x = unicodedata.normalize("NFD", x.lower()).replace("đ", "d")
    return "".join(c for c in x if unicodedata.category(c) != "Mn")
def cham(muc_list, dau_ra):
    trong = [(m, o) for m, o in zip(muc_list, dau_ra) if not m["ngoai_kho"]]
    ngoai = [(m, o) for m, o in zip(muc_list, dau_ra) if m["ngoai_kho"]]
    def trich(o): return {int(x) for x in re.findall(r"\[(\d+)\]", o)}
    co = sum(bool(trich(o)) for _, o in trong)
    hl = sum(bool(trich(o)) and trich(o) <= {1, 2, 3} for _, o in trong)
    sach = [(m, o) for m, o in ngoai if m["hoi"] not in AM_KHO_RO_RI]
    ban = [(m, o) for m, o in ngoai if m["hoi"] in AM_KHO_RO_RI]
    tc = sum("khong tim thay" in _chuan(o) for _, o in sach)
    tc_nham = sum("khong tim thay" in _chuan(o) for _, o in trong)
    return dict(co_trich_dan=f"{co}/{len(trong)}", trich_dan_hop_le=f"{hl}/{len(trong)}",
                tu_choi_dung_ngoai_kho_sach=f"{tc}/{len(sach)}", tu_choi_nham_trong_kho=f"{tc_nham}/{len(trong)}",
                cau_am_kho_ro_ri_tu_choi=[("khong tim thay" in _chuan(o)) for _, o in ban])
def nap(): return json.load(open(NGU_CANH, encoding="utf-8"))
