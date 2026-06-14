import pickle
from entity.instruction import Instruction
from objectives.objective import Objective

plm = "google/flan-t5-base"

# All 3 tasks matching the paper (Laptop14=ABSA, SST2=TC, SNLI=NLI)
TASKS = [
    {"dataset": "Laptop14", "output_file": "noinstruct_laptop14.pkl"},
    {"dataset": "SST2",     "output_file": "noinstruct_sst2.pkl"},
    {"dataset": "SNLI",     "output_file": "noinstruct_snli.pkl"},
]

if __name__ == "__main__":
    all_results = {}

    for task in TASKS:
        dataset = task["dataset"]
        output_file = task["output_file"]

        print("=" * 80)
        print(f"Running Baseline: NoInstruct | Dataset: {dataset} | PLM: {plm}")
        print("=" * 80)

        # NoInstruct: empty definition and example
        inst = Instruction(definition="", example="", dataset=dataset)

        # Each instruction class sets a default eos_instruction even when bos/example=""
        # (APCInstruction: "let us predict sentiments one by one: "
        #  TCInstruction:  "The statement is:"
        #  NLIInstruction: "Your judgment is:")
        # Clear it so the model receives no structural cue from the instruction.
        inst.prompt.eos_instruction = ""

        print(f"Instruction length after clearing: {len(inst.prompt)}")
        print("Evaluating...")

        obj = Objective(inst_individual=inst, dataset=dataset, plm=plm)

        try:
            reciprocal_metric = float(obj.objectives[2])
            avg_metric = (1.0 / reciprocal_metric) / 4.0 if reciprocal_metric != 0 else 0
        except Exception:
            avg_metric = 0

        res = {
            "dataset":     dataset,
            "instruction": "",
            "example":     "",
            "objectives":  list(obj.objectives),
            "metrics":     obj.metric,
            "avg_metric":  avg_metric,
        }

        print(f"\nNoInstruct Results [{dataset}]:")
        if isinstance(obj.metric, dict):
            for k, v in obj.metric.items():
                print(f"  {k}: {v * 100:.2f}%")
        print(f"  Average metric: {avg_metric * 100:.2f}%")

        pickle.dump([res], open(output_file, "wb"))
        print(f"Saved to: {output_file}\n")

        all_results[dataset] = res

    # Save combined results for easy comparison
    pickle.dump(all_results, open("noinstruct_all.pkl", "wb"))

    print("=" * 80)
    print("All NoInstruct tasks done. Summary:")
    for ds, res in all_results.items():
        print(f"  {ds}: avg_metric={res['avg_metric']*100:.2f}%")
    print("Combined results saved to: noinstruct_all.pkl")
    print("=" * 80)
