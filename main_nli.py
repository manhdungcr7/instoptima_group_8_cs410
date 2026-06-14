import pickle
from evo_core.nsga2 import nsga2

# 1. Cấu hình test siêu nhỏ cho Laptop
generation_num = 10
population_size = 30

dataset = "SNLI"

# Dùng bản small để test cho nhẹ máy
plm = "google/flan-t5-base"

print(f"Bắt đầu chạy test task NLI với dataset: {dataset}...")

evo_data = nsga2(
    population_size,
    num_generations=generation_num,
    dataset=dataset,
    plm=plm,
)

# 2. Đổi tên file lưu để không đè lên các task khác
print("\nĐang lưu kết quả...")
pickle.dump(evo_data, open("nsga2_result_nli.pkl", "wb"))

try:
    pareto_front = evo_data["fronts"][0]
    print("\n" + "="*50)
    print("KẾT QUẢ PARETO FRONT ĐẦU TIÊN:")
    print("="*50)
    for individual in pareto_front:
        print("-" * 100)
        print(f"Prompt: {individual.instruction}")
        print(f"Objectives: {individual.objectives}")
except Exception as e:
    print(f"Lỗi in kết quả: {e}")

print("Hoàn tất chạy thử NLI!")