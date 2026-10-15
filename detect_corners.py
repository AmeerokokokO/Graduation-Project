# -*- coding: utf-8 -*-
import cv2             # مكتبة OpenCV للصور والفيديو
import os              # للتعامل ويا الملفات والمسارات
import glob            # حتى نقرأ مجموعة ملفات من فولدر
import argparse        # حتى ناخذ باراميترات من سطر الأوامر
import uuid            # حتى نولد أسماء عشوائية للصور اللي نريد نحفظها
from corners import harris_corners, shi_tomasi, canny_edges, titled  # نستورد الدوال من الملف الثاني
import numpy as np     # للعمليات الرياضية و الـ arrays

# ---------------------- تحسين الإضاءة ----------------------
def adaptive_gamma(img):
    # نحول الصورة من BGR الى  حتى نشتغل بس على قناة الإضاءة Y
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # نفصل القنوات
    y, cr, cb = cv2.split(ycrcb)

    # نحسب متوسط الإضاءة للصورة
    mean = np.mean(y)

    # نحدد قيمة جاما حسب الإضاءة: اذا الصورة غامقة نرفع الجاما، والعكس
    gamma = 1.5 - (mean / 255.0)
    # نثبت القيمة ضمن مدى [0.5, 1.8] حتى ما تبالغ بالتعديل
    gamma = max(0.5, min(gamma, 1.8))

    # نطبق تصحيح جاما على قناة Y
    y_corr = ((y / 255.0) ** gamma) * 255.0
    # نتأكد القيم تبقى بين 0 و 255 ونحوّلها لـ uint8
    y_corr = np.clip(y_corr, 0, 255).astype(np.uint8)

    # نرجع ندمج القنوات Y و Cr و Cb من جديد
    out = cv2.merge([y_corr, cr, cb])
    # نرجع الصورة الى BGR حتى نكمل شغلنا الطبيعي
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def apply_clahe_rgb(img, clip=2.0, grid=8):
    # نحول الصورة من BGR الى LAB حتى نشتغل على قناة الإضاءة L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # نفصل القنوات
    l, a, b = cv2.split(lab)

    # ننشئ كائن CLAHE بقيمة clipLimit وحجم grid
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    # نطبق CLAHE على قناة L
    l2 = clahe.apply(l)

    # نرجع ندمج L المعدلة ويا باقي القنوات
    lab2 = cv2.merge([l2, a, b])
    # نرجع الصورة الى BGR
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# ---------------------- أدوات ترتيب الصور ----------------------
def resize_exact(img, w=480, h=360):
    # نغير حجم الصورة لحجم ثابت (عرض w، ارتفاع h)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def pad(img, px=8, color=(30, 30, 30)):
    # نضيف إطار (Border) حول الصورة بعدد بكسلات px وباللون المحدد
    return cv2.copyMakeBorder(
        img, px, px, px, px,
        cv2.BORDER_CONSTANT,
        value=color
    )

def grid2x2(a, b, c, d, tile_w=480, tile_h=360, pad_px=8, bg=(30, 30, 30)):
    # نجهز كل صورة: نغير حجمها ونضيف لها إطار
    a = pad(resize_exact(a, tile_w, tile_h), pad_px, bg)
    b = pad(resize_exact(b, tile_w, tile_h), pad_px, bg)
    c = pad(resize_exact(c, tile_w, tile_h), pad_px, bg)
    d = pad(resize_exact(d, tile_w, tile_h), pad_px, bg)

    # نلصق صورتين أفقياً: فوق
    top = cv2.hconcat([a, b])
    # نلصق صورتين أفقياً: جوه
    bottom = cv2.hconcat([c, d])
    # نلصق الصفين عمودياً حتى تطلع شبكة 2x2
    return cv2.vconcat([top, bottom])

def grid2x3(a, b, c, d, e, f, tile_w=480, tile_h=360, pad_px=8, bg=(30, 30, 30)):
    # نجهز كل صورة: نغير حجمها ونضيف لها إطار (6 صور)
    a = pad(resize_exact(a, tile_w, tile_h), pad_px, bg)
    b = pad(resize_exact(b, tile_w, tile_h), pad_px, bg)
    c = pad(resize_exact(c, tile_w, tile_h), pad_px, bg)
    d = pad(resize_exact(d, tile_w, tile_h), pad_px, bg)
    e = pad(resize_exact(e, tile_w, tile_h), pad_px, bg)
    f = pad(resize_exact(f, tile_w, tile_h), pad_px, bg)

    # نلصق ثلاث صور أفقياً: الصف الأول
    top = cv2.hconcat([a, b, c])
    # نلصق ثلاث صور أفقياً: الصف الثاني
    bottom = cv2.hconcat([d, e, f])
    # نلصق الصفين عمودياً حتى تطلع شبكة 2x3
    return cv2.vconcat([top, bottom])

def save_panel(panel, out_dir, stem):
    # نتأكد الفولدر موجود، واذا مو موجود ينشئه
    os.makedirs(out_dir, exist_ok=True)
    # نبني المسار الكامل لملف الإخراج
    path = os.path.join(out_dir, f"{stem}.png")
    # نحفظ الصورة على القرص
    cv2.imwrite(path, panel)
    # نرجع المسار حتى نطبعه
    return path

# ---------------------- مرحلة المعالجة ----------------------
def process_image(img, args):

    # 1) تحسين الإضاءة باستخدام Adaptive Gamma اذا المستخدم مفعّله
    if args.use_gamma:
        img = adaptive_gamma(img)

    # 2) تطبيق CLAHE على الصورة الملونة (LAB) اذا مفعّل
    if args.use_clahe:
        img = apply_clahe_rgb(img, clip=args.clahe_clip, grid=args.clahe_grid)

    # 3) نطبق Harris ونرجع الصور الثلاثة الخاصة بيه
    h_overlay, h_heat, _h_sil = harris_corners(
        img,
        block_size=args.h_block,
        ksize=args.h_ksize,
        k=args.h_k,
        thresh=args.h_thresh,
        dilate_iter=args.h_dilate,
        colormap=args.h_colormap
    )

    # 4) نطبق Shi–Tomasi ونرجع صورتين
    s_overlay, s_sil = shi_tomasi(
        img,
        max_corners=args.s_max,
        quality_level=args.s_q,
        min_distance=args.s_min,
        block_size=args.s_block,
        radius=args.s_radius
    )

    # 5) نطبق Canny Edge Detection ونرجع ثلاث صور
    c_edges, c_colored, c_overlay = canny_edges(
        img,
        blur_ksize=args.c_blur_ksize,
        blur_sigma=args.c_blur_sigma,
        threshold1=args.c_thresh1,
        threshold2=args.c_thresh2
    )
    # نحول الحواف من grayscale لـ BGR حتى نكدر نعرضها بالشبكة
    c_edges_bgr = cv2.cvtColor(c_edges, cv2.COLOR_GRAY2BGR)

    # 6) نضيف عناوين لكل صورة باستخدام titled
    p1 = titled(h_overlay, "Harris • Overlay")
    p2 = titled(h_heat,    "Harris • Heatmap")
    p3 = titled(c_overlay, "Canny • Overlay")
    p4 = titled(s_overlay, "Shi–Tomasi • Overlay")
    p5 = titled(s_sil,     "Shi–Tomasi • Silhouette")
    p6 = titled(c_edges_bgr, "Canny • Edges")

    # نجمع الست صور في لوحة وحدة 2x3
    panel = grid2x3(p1, p2, p3, p4, p5, p6,
                    tile_w=args.tile_w,
                    tile_h=args.tile_h,
                    pad_px=args.pad)

    # نرجع اللوحة الجاهزة للعرض او الحفظ
    return panel

# ---------------------- تشغيل البرنامج ----------------------
def run_single(path, args):
    # نقرأ صورة وحدة من المسار
    img = cv2.imread(path)
    # نعالج الصورة ونطلع لوحة كاملة
    panel = process_image(img, args)
    # نعرض اللوحة في نافذة
    cv2.imshow("Corners Dashboard", panel)
    # ننتظر من المستخدم يضغط مفتاح (0 يعني ينتظر الى ما يضغط شي)
    key = cv2.waitKey(0) & 0xFF
    # اذا مفعّل الحفظ او المستخدم ضغط s نحفظ الصورة
    if args.save or key == ord('s'):
        stem = os.path.splitext(os.path.basename(path))[0]
        saved = save_panel(panel, args.output, stem)
        print(f"Saved: {saved}")
    # نغلق كل النوافذ
    cv2.destroyAllWindows()

def run_folder(folder, args):
    # الامتدادات اللي راح نبحث عنها داخل الفولدر
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    # نلف عالامتدادات ونضيف كل الملفات الموجودة
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    # نرتب الملفات حتى يكون العرض مرتب
    files.sort()
    # نلف على كل صورة داخل الفولدر
    for p in files:
        img = cv2.imread(p)
        panel = process_image(img, args)
        cv2.imshow("Corners Dashboard", panel)
        # ننتظر شوي (1ms) ونقرأ الزر المضغوط
        key = cv2.waitKey(1) & 0xFF
        # اذا الحفظ مفعّل او المستخدم ضغط s نحفظ النتيجة
        if args.save or key == ord('s'):
            stem = os.path.splitext(os.path.basename(p))[0]
            save_panel(panel, args.output, stem)
        # اذا ضغط q نطلع من اللوب
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def run_camera(cam_idx, args):
    # نفتح الكاميرا حسب رقمها
    cap = cv2.VideoCapture(cam_idx)
    # اذا ما انفتح، نطبع رسالة ونطلع
    if not cap.isOpened():
        print("Camera open failed")
        return
    # لوب قراءة مستمرة من الكاميرا
    while True:
        ok, frame = cap.read()
        # اذا ما قدر يقرأ فريم نكسر اللوب
        if not ok:
            break
        # نعالج الفريم ونحول الى لوحة
        panel = process_image(frame, args)
        # نعرض اللوحة
        cv2.imshow("Corners Dashboard", panel)
        # نقرأ الزر المضغوط
        key = cv2.waitKey(1) & 0xFF
        # اذا الحفظ مفعّل والمستخدم ضغط s نحفظ صورة بلوحة كاملة باسم عشوائي
        if args.save and key == ord('s'):
            save_panel(panel, args.output, str(uuid.uuid4())[:8])
        # اذا ضغط q نوقف
        if key == ord('q'):
            break
    # نغلق الكاميرا
    cap.release()
    # نغلق النوافذ
    cv2.destroyAllWindows()

# ---------------------- إعدادات المستخدم ----------------------
def parse_args():
    # نستخدم argparse حتى ناخذ خيارات من سطر الاوامر
    ap = argparse.ArgumentParser()

    # أنواع الإدخال:
    # 0 = صورة وحدة، 1 = كاميرا، 2 = فولدر صور
    ap.add_argument("--input_type", type=int, default=0)
    # مسار الصورة اذا نستخدم input_type=0
    ap.add_argument("--img_path", type=str, default="")
    # رقم الكاميرا اذا نستخدم input_type=1
    ap.add_argument("--camera", type=int, default=0)
    # مسار الفولدر اذا نستخدم input_type=2
    ap.add_argument("--folder_path", type=str, default="images")
    # خيار الحفظ (true/false) نتحول من نص الى Boolean
    ap.add_argument("--save",
                    type=lambda x: str(x).lower()=="true",
                    default=False)
    # فولدر الإخراج اللي نخزن بيه اللوحات
    ap.add_argument("--output", type=str, default="output")

    # إعدادات حجم اللوحة Tiles
    ap.add_argument("--tile_w", type=int, default=480)
    ap.add_argument("--tile_h", type=int, default=360)
    ap.add_argument("--pad",    type=int, default=8)

    # تحسين الإضاءة:
    # تفعيل/تعطيل Adaptive Gamma
    ap.add_argument("--use_gamma",
                    type=lambda x: str(x).lower()=="true",
                    default=True)
    # تفعيل/تعطيل CLAHE على الصورة الملونة
    ap.add_argument("--use_clahe",
                    type=lambda x: str(x).lower()=="true",
                    default=True)
    # قوة الـ CLAHE
    ap.add_argument("--clahe_clip", type=float, default=40.0)
    # حجم grid مال الـ CLAHE
    ap.add_argument("--clahe_grid", type=int, default=8)

    # باراميترات Harris
    ap.add_argument("--h_block",   type=int,   default=5)
    ap.add_argument("--h_ksize",   type=int,   default=3)
    ap.add_argument("--h_k",       type=float, default=0.01)
    ap.add_argument("--h_thresh",  type=float, default=0.01)
    ap.add_argument("--h_dilate",  type=int,   default=1)
    ap.add_argument("--h_colormap",type=int,   default=cv2.COLORMAP_TURBO)

    # باراميترات Shi–Tomasi
    ap.add_argument("--s_max",    type=int,   default=500)
    ap.add_argument("--s_q",      type=float, default=0.1)
    ap.add_argument("--s_min",    type=int,   default=10)
    ap.add_argument("--s_block",  type=int,   default=3)
    ap.add_argument("--s_radius", type=int,   default=3)

    # باراميترات Canny Edge Detection
    ap.add_argument("--c_blur_ksize", type=int,   default=21)    # حجم فلتر Gaussian (لازم فردي)
    ap.add_argument("--c_blur_sigma", type=float, default=1.4)  # سيجما للـ Blur
    ap.add_argument("--c_thresh1",    type=int,   default=100)  # العتبة الأولى
    ap.add_argument("--c_thresh2",    type=int,   default=200)  # العتبة الثانية

    # نرجع الـ args حتى نستخدمها بباقي الفنكشنات
    return ap.parse_args()

# ---------------------- نقطة البداية ----------------------
def main():
    # نقرأ كل الإعدادات من سطر الأوامر
    args = parse_args()

    # اذا input_type = 0 ونطينا img_path نشتغل على صورة وحدة
    if args.input_type == 0 and args.img_path:
        run_single(args.img_path, args)
    # اذا input_type = 1 نستخدم الكاميرا
    elif args.input_type == 1:
        run_camera(args.camera, args)
    # اذا input_type = 2 نشتغل على فولدر كامل
    elif args.input_type == 2:
        run_folder(args.folder_path, args)
    # غيرها يعني المستخدم مدخل شي غلط
    else:
        print("Invalid or missing input, see README usage.")

# اذا الملف هو الرئيسي اللي يشتغل، نفذ main
if __name__ == "__main__":
    main()