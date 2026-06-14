import pickle

def safe_analyze(pkl_path):
    print(f"\n{'='*70}")
    print(f"🔍 PHÂN TÍCH FILE: {pkl_path}")
    print(f"{'='*70}")
    
    try:
        evo_data = pickle.load(open(pkl_path, "rb"))
    except Exception as e:
        print(f"Lỗi mở file: {e}")
        return
        
    population = evo_data.get('population', []) if isinstance(evo_data, dict) else evo_data
    
    if not hasattr(population, '__iter__'):
        population = [population]
        
    for i, ind in enumerate(population):
        prompt_text = "Không tìm thấy Prompt"
        if hasattr(ind, 'genotype'):
            prompt_text = str(ind.genotype)
        elif hasattr(ind, 'instruction') and hasattr(ind.instruction, 'definition'):
            prompt_text = str(ind.instruction.definition)
        else:
            prompt_text = str(ind)
            
        if "I charge it at night" in prompt_text:
            prompt_text = prompt_text.split("I charge it at night")[0].strip()
            
        metrics = "Chưa có điểm"
        if hasattr(ind, 'objectives'):
            metric_val = getattr(ind.objectives, 'metric', None)
            if metric_val and isinstance(metric_val, list) and len(metric_val) >= 2:
                # In ra điểm số chuẩn
                metrics = f"Accuracy: {metric_val[0]:.4f} | Length: {metric_val[1]:.4f}"
            else:
                metrics = str(getattr(ind, 'objectives', ''))

        print(f"👉 [Cá thể {i}]")
        print(f"📝 Prompt: {prompt_text.replace(chr(10), ' ')}")
        print(f"📊 Điểm: {metrics}\n")

# Chỉ đọc đúng file kết quả cuối cùng
safe_analyze("nsga2_result_v4.pkl")