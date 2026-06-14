from evo_core.nsga2 import nsga2
import pickle # Thêm thư viện này để lưu file giống file main.py

# Configurations
# 1. Hạ số thế hệ xuống 1 (chỉ cần sinh ra con và chạy thử 1 vòng)
generation_num = 10

# 2. Hạ số cá thể xuống 2 (số lượng nhỏ nhất để NSGA-II có thể lai ghép)
population_size = 30 
tournament_size = 2

dataset = "SST2"

# 3. CHÚ Ý: Hạ model xuống small để laptop chạy được
# "base" khá nặng, nếu laptop không có GPU mạnh sẽ bị tràn RAM hoặc chạy rất lâu.
# plm = "google/flan-t5-base" 
plm = "google/flan-t5-base" 

evo_data = nsga2(
    population_size,
    num_generations=generation_num,
    tournament_size=tournament_size,
    dataset=dataset,
    plm=plm,
)

# 4. Thêm đoạn code lưu file và in kết quả giống main.py 
# (Vì code gốc tác giả chỉ print(evo_data) rất sơ sài)
print("\nĐang lưu kết quả...")
pickle.dump(evo_data, open("nsga2_result_tc.pkl", "wb"))

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
    print(f"Chưa in được Pareto Front: {e}")

print("Hoàn tất chạy thử!")