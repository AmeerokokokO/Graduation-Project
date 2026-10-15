# Graduation Project: Corner & Edge Detection (مشروع التخرج)

نظام متقدم لمعالجة الصور وكشف الحواف والزوايا باستخدام خوارزميات الرؤية الحاسوبية (Computer Vision) ومكتبة OpenCV.

---

## 📌 الميزات الرئيسية (Features)
- **خوارزميات كشف الزوايا والحواف:**
  - Harris Corner Detection
  - Shi-Tomasi Corner Detection
  - Canny Edge Detection
- **تحسين الإضاءة ومعالجة الصور المسبقة:**
  - تحسين الإضاءة التكيفي (Adaptive Gamma Correction)
  - تحسين التباين الموضعي المتقدم (CLAHE)
- **أنماط إدخال متعددة:**
  1. معالجة صورة واحدة محددة (`--input_type=0`)
  2. تشغيل الكاميرا في الوقت الحقيقي (Live Webcam) (`--input_type=1`)
  3. معالجة مجلد صور كامل دفعة واحدة (`--input_type=2`)
- **حفظ المخرجات:** إمكانية حفظ النتائج المرئية مع إطارات وعناوين مقارنة واضحة.

---

## 🚀 متطلبات التشغيل (Requirements)

تأكد من تثبيت بايثون (Python 3.8+) ثم قم بتثبيت المكتبات المطلوبة:

```bash
pip install -r requirements.txt
```

---

## 💻 طريقة الاستخدام (How to Run)

### 1. تشغيل الكاميرا الحية (Live Camera)
```bash
python detect_corners.py --input_type=1 --camera=0
```

### 2. معالجة صورة واحدة وحفظ النتيجة
```bash
python detect_corners.py --input_type=0 --img_path=imges/img2.png --save=True --output=output
```

### 3. معالجة مجلد صور كامل
```bash
python detect_corners.py --input_type=2 --folder_path=imges7 --save=True --output=output7
```

---

## 📁 هيكلية المشروع (Project Structure)
- `detect_corners.py`: البرنامج الرئيسي لمعالجة الإدخال وتشغيل خوارزميات الكشف والعرض.
- `corners.py`: الدوال الرياضية وتطبيق خوارزميات Harris, Shi-Tomasi, Canny.
- `requirements.txt`: متطلبات ومكتبات المشروع.
- `imges/`, `imges2/`, ...: مجلدات تحتوي على صور العينات والاختبار.
- `output/`, `output2/`, ...: مجلدات حفظ النتائج المعالجة.
