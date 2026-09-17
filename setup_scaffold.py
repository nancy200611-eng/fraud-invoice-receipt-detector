import os

base_dir = r"E:\fraud Invoice&Receipt Detector"
dirs = [
    "data/raw",
    "data/ocr_extracted",
    "data/manipulated/images",
    "data/manipulated/masks",
    "data/samples",
    "src/ocr",
    "src/manipulation",
    "src/forensics",
    "src/models",
    "src/rules",
    "src/backend",
    "src/backend/static",
    "tests"
]
for d in dirs:
    p = os.path.join(base_dir, d)
    os.makedirs(p, exist_ok=True)
    if not d.startswith("data") and not d.endswith("static"):
        init_py = os.path.join(p, "__init__.py")
        if not os.path.exists(init_py):
            with open(init_py, "w", encoding="utf-8") as f:
                f.write("# init\n")

src_init = os.path.join(base_dir, "src", "__init__.py")
with open(src_init, "w", encoding="utf-8") as f:
    f.write('__version__ = "1.0.0"\n')

print("Directory structure initialized successfully.")
