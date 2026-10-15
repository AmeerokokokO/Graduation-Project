
import cv2          # نستورد مكتبة OpenCV حتى نتعامل ويا الصور
import numpy as np  # نستورد Numpy حتى نشتغل عالـ arrays والعمليات الرياضية


def _to_uint8(img):
    # هنانا نسوي Normalize للصورة حتى نحول مدى القيم من 0 لـ 255
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # وبعدين نحول نوع الصورة الى 8-bit حتى نكدر نعرضها ونطبق عليها colormap
    return norm.astype(np.uint8)


def _title_bar(img, title, h=40, bg=(24, 24, 24), fg=(255, 255, 255)):
    # هنانا نسوي مستطيل فوق الصورة نكتب بيه عنوان 
    # نسوي بار ارتفاعه h وعرضه نفس عرض الصورة، ولونه 
    bar = np.full((h, img.shape[1], 3), bg, dtype=np.uint8)
    # نكتب النص مال العنوان على البار بمكان ثابت تقريبا
    cv2.putText(
        bar,                  # الصورة اللي راح نكتب عليها
        title,                # النص
        (12, int(h * 0.7)),   # موقع الكتابة (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # نوع الخط
        0.8,                  # حجم الخط
        fg,                   # لون الخط
        2,                    # سماكة الخط
        cv2.LINE_AA           # نوع الرسم (Anti-Aliased)
    )
    # نلصق البار فوق الصورة الأصلية (Stack عمودي)
    return np.vstack([bar, img])

# ---------- Harris ----------
def harris_corners(
        image,
        block_size=3,      # حجم البلوك اللي نحسب عليه المشتقات
        ksize=3,           # حجم فلتر Sobel
        k=0.04,            # ثابت Harris
        thresh=0.01,       # العتبة اللي نختار منها النقاط
        dilate_iter=1,     # كم مرة نكرر عملية الـ dilation
        colormap=cv2.COLORMAP_TURBO  # نوع الـ colormap للـ heatmap
    ):
    # ناخذ نسخة من الصورة حتى ما نعدّل عالأصلي
    img = image.copy()

    # نحول الصورة من BGR لـ Gray حتى نطبق Harris (لازم grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # هنانا نستخدم CLAHE حتى نزيد التباين قبل Harris (مفيد للـ Low Contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # نحدد قوة القص وحجم التايلز
    gray = clahe.apply(gray)  # نطبق CLAHE على الصورة الرمادية

    # نحول الصورة الرمادية الى float32 لأن cornerHarris يحتاج هذا النوع
    gray_f32 = np.float32(gray)

    # نطبّق خوارزمية Harris حتى نحسب استجابة الزوايا لكل بكسل
    dst = cv2.cornerHarris(gray_f32, block_size, ksize, k)

    # اذا نريد نكبر النقاط (نبرزها) نستخدم dilation
    if dilate_iter > 0:
        dst = cv2.dilate(dst, None, iterations=dilate_iter)

    # نسوي mask للنقاط اللي قيمتها أكبر من نسبة thresh من أعلى قيمة
    mask = dst > (thresh * dst.max())

    # نحول dst الى 8-bit حتى نكدر نطبق عليها colormap
    heat_u8 = _to_uint8(dst)
    # نطبّق colormap حتى يطلع شكل heatmap يوضح وين القيم العالية
    heat_colored = cv2.applyColorMap(heat_u8, colormap)
    # ندمج الصورة الأصلية ويا الـ heatmap بنسب معينة حتى تطلع Overlay
    overlay = cv2.addWeighted(img, 0.7, heat_colored, 0.3, 0.0)
    # البكسلات اللي mask مالته True نخليها لون أصفر (زاوية)
    overlay[mask] = (0, 255, 255)

    # نسوي صورة سودة نفس حجم الصورة ونخلي بس الزوايا باللون الأبيض (Silhouette)
    silhouette = np.zeros_like(img)
    silhouette[mask] = (255, 255, 255)

    # نرجع 3 صور: الـ Overlay، والـ Heatmap الملوّنة، والـ Silhouette
    return overlay, heat_colored, silhouette


# ---------- Shi–Tomasi ----------
def shi_tomasi(
        image,
        max_corners=600,     # أقصى عدد زوايا نسمح بيها
        quality_level=0.01,  # جودة أقل زاوية مقبولة (نسبة من أفضل واحدة)
        min_distance=10,     # أقل مسافة بين نقطة والثانية
        block_size=3,        # حجم البلوك لحساب الكوفارينس
        radius=3,            # نصف قطر الدائرة اللي نرسمها
        color=(0, 255, 255)  # لون الدوائر مال النقاط
    ):

    # ناخذ نسخة من الصورة حتى ما نخرب الأصلية
    img = image.copy()

    # 1) نحول الصورة لـ grayscale حتى نشتغل على قناة وحدة
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) نستخدم CLAHE حتى نقوّي التباين خصوصاً بالصور اللي بيها إضاءة ضعيفة
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) نستخدم دالة goodFeaturesToTrack اللي تطبّق Shi–Tomasi تلقائياً
    pts = cv2.goodFeaturesToTrack(
        gray,                  # الصورة الرمادية
        maxCorners=max_corners,   # عدد النقاط الأقصى
        qualityLevel=quality_level,  # أقل جودة مقبولة
        minDistance=min_distance,    # أقل مسافة بين نقطة والثانية
        blockSize=block_size,       # حجم البلوك
        useHarrisDetector=False     # False يعني نستخدم Shi–Tomasi مو Harris
    )

    # نسوي صورة سودة حتى نخزن بيها بس نقاط الزوايا
    silhouette = np.zeros_like(img)

    # اذا فعلاً انكَتشف نقاط
    if pts is not None:
        # نحول النقاط الى أعداد صحيحة (int)
        pts = np.intp(pts)
        # نلف عكل نقطة
        for p in pts:
            x, y = int(p[0][0]), int(p[0][1])  # احداثيات النقطة
            # نرسم دائرة صغيرة على مكان الزاوية بالصورة الملونة
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            # ونرسم نفس الدائرة بس باللون الأبيض على صورة الـ silhouette
            cv2.circle(silhouette, (x, y), radius, (255, 255, 255), -1, cv2.LINE_AA)

    # نرجع صورتين: وحدة عليها النقاط، ووحدة Silhouette بس ابيض وأسود
    return img, silhouette


# ---------- Canny Edge Detection ----------
def canny_edges(
        image,
        blur_ksize=5,        # حجم فلتر Gaussian Blur (لازم يكون فردي)
        blur_sigma=1.4,      # قيمة السيجما للـ Blur
        threshold1=100,      # العتبة الأولى للـ Canny
        threshold2=200,      # العتبة الثانية للـ Canny
        edge_color=(0, 255, 0)  # لون الحواف (أخضر افتراضياً)
    ):
    # ناخذ نسخة من الصورة حتى ما نعدّل عالأصلي
    img = image.copy()

    # 1) نحول الصورة من BGR لـ Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) نطبق Gaussian Blur حتى نشيل الضوضاء قبل الـ Canny
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

    # 3) نطبق خوارزمية Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # 4) نسوي صورة ملونة للعرض: خلفية سوداء والحواف بلون معين
    edges_colored = np.zeros_like(img)
    edges_colored[edges > 0] = edge_color

    # 5) نسوي overlay: ندمج الصورة الأصلية ويا الحواف
    overlay = img.copy()
    overlay[edges > 0] = edge_color

    # نرجع 3 صور: الحواف بالأبيض، الحواف ملونة، والـ Overlay
    return edges, edges_colored, overlay


# ---------- Titles ----------
def titled(panel, title):
    # بس نضيف شريط عنوان فوق الصورة باستخدام الهيلبر اللي فوق
    return _title_bar(panel, title)
