import pickle

# 1. Đổi tên file này thành file pkl em muốn kiểm tra (ví dụ SST2 hoặc Laptop14)
file_path = "nsga2_result_v4.pkl" 

print("=" * 80)
print(f"🔍 ĐANG TRÍCH XUẤT CÁC PROMPT TỐT NHẤT TỪ: {file_path}")
print("=" * 80)

try:
    # Đọc dữ liệu tiến hóa
    evo_data = pickle.load(open(file_path, "rb"))
    
    # Lấy Front 0 (Mặt cắt Pareto chứa những cá thể xuất sắc nhất thế hệ cuối)
    pareto_front = evo_data["fronts"][0]
    print(f"✨ Tìm thấy {len(pareto_front)} câu lệnh tối ưu trong Pareto Front 0.\n")

    # Sắp xếp các cá thể ưu tiên theo Accuracy. 
    # Trong code của em, objectives[2] là reciprocal_metric (Nghịch đảo của điểm số). 
    # Số này CÀNG NHỎ thì Accuracy CÀNG CAO. Nên ta sort tăng dần theo objectives[2].
    sorted_front = sorted(pareto_front, key=lambda x: float(x.objectives[2]))

    # In ra Top 5 câu lệnh đỉnh nhất
    for i, ind in enumerate(sorted_front[:5]):
        length = float(ind.objectives[0])
        perplexity = float(ind.objectives[1])
        recip = float(ind.objectives[2])
        
        print(f"🏆 TOP {i+1}:")
        print(f"📝 Lệnh (Prompt): {ind.instruction}")
        print(f"📊 Objectives : Length = {length} | Perplexity = {perplexity:.2f} | Reciprocal = {recip:.4f}")
        print("-" * 80)

except FileNotFoundError:
    print(f"❌ Không tìm thấy file {file_path}. Hãy kiểm tra lại tên file!")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")