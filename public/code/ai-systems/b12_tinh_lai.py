"""B12 - tinh lai thuoc do chat luong tu ket qua da luu, tach rieng cau am kho bi ro ri.
Khong can chay lai mo hinh.  Dung:  python b12_tinh_lai.py <thu_muc_ket_qua>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b12_chung as C
K = sys.argv[1] if len(sys.argv) > 1 else "/media/ai/SSD/vietpv/booksys/ket-qua"
muc = json.load(open(os.path.join(K, "b12-ngucanh.json"), encoding="utf-8"))["muc"]
tr = [m["cos_dau"] for m in muc if not m["ngoai_kho"]]
ng = [m["cos_dau"] for m in muc if m["ngoai_kho"] and m["hoi"] not in C.AM_KHO_RO_RI]
print("diem truy hoi: trong kho thap nhat %.4f | ngoai kho sach cao nhat %.4f | tach sach: %s" % (min(tr), max(ng), max(ng) < min(tr)))
for c in ("A", "B", "C", "D", "F0", "F"):
    d = json.load(open(os.path.join(K, f"b12-{c}.json"), encoding="utf-8"))
    print(c, C.cham(muc, d["dau_ra"]))
