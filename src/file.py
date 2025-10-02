import torch
import faiss

# ------------------------------
# 1. Перевірка Torch / GPU
# ------------------------------
gpu_available = torch.cuda.is_available()
print("Torch CUDA available:", gpu_available)

if gpu_available:
    print("Torch device:", torch.cuda.get_device_name(0))
    # створимо тензор на GPU
    x = torch.randn(3, 3).cuda()
    print("Tensor on GPU:", x.device)
else:
    print("GPU недоступний, Torch буде використовувати CPU")

# ------------------------------
# 2. Перевірка FAISS
# ------------------------------
try:
    # CPU варіант (працює на Windows)
    index = faiss.IndexFlatL2(10)
    print("FAISS CPU OK ✅")

    # GPU варіант (тільки Linux/macOS, не працює на Windows)
    try:
        res = faiss.StandardGpuResources()
        index_gpu = faiss.IndexFlatL2(10)
        print("FAISS GPU OK ✅")
    except Exception:
        print("FAISS GPU недоступний на цій системі")
except Exception as e:
    print("FAISS не встановлений або помилка:", e)

# ------------------------------
# 3. Перевірка можливості навчання нейронки на GPU
# ------------------------------
if gpu_available:
    # невеликий приклад навчання
    model = torch.nn.Linear(5, 2).cuda()
    input = torch.randn(10, 5).cuda()
    output = model(input)
    print("Нейронка працює на GPU:", output.device)
else:
    print("Нейронка працюватиме на CPU")
