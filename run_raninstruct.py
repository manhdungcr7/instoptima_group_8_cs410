import time
import json
import pickle

from chatgpt import Chatbot
from entity.instruction import Instruction
from objectives.objective import Objective

plm = "google/flan-t5-base"
NUM_MUTATIONS = 5   # paper: "generates five random instructions"
MAX_RETRIES = 3

# All 3 tasks matching the paper (Laptop14=ABSA, SST2=TC, SNLI=NLI)
TASKS = [
    {
        "dataset":     "Laptop14",
        "task_desc":   "aspect-based sentiment analysis",
        "output_file": "raninstruct_laptop14.pkl",
    },
    {
        "dataset":     "SST2",
        "task_desc":   "text classification (sentiment analysis)",
        "output_file": "raninstruct_sst2.pkl",
    },
    {
        "dataset":     "SNLI",
        "task_desc":   "natural language inference",
        "output_file": "raninstruct_snli.pkl",
    },
]


def generate_random_instruction(bot, base_def, base_ex, task_desc, attempt_label):
    """Call ChatGPT to generate one random (definition, example) pair."""
    prompt = f"""I want you to be a professional prompt engineer. \
I am working on multi-objective evolutionary prompt optimization for {task_desc} \
and need your help to design diverse instruction templates.

Here is the original instruction definition:
{base_def}

Here is the original instruction example:
{base_ex}

Please generate a new, paraphrased, or mutated version of this instruction \
(both definition and example) to create diversity.
CRITICAL RULES:
1. The example MUST keep the same label format and structure as the original, \
so downstream code can parse it correctly.
2. Reply with ONLY a single valid JSON object with exactly two keys: \
"definition" and "example".
3. Do not include any other text, explanations, or markdown wrappers.
"""
    for attempt in range(MAX_RETRIES):
        try:
            response = bot.chat(prompt)
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:].rstrip("`").strip()
            elif clean.startswith("```"):
                clean = clean[3:].rstrip("`").strip()
            parsed = json.loads(clean)
            return parsed["definition"], parsed["example"]
        except Exception as e:
            print(f"    Parse error on attempt {attempt + 1}: {e}")
            time.sleep(2)

    print(f"    All retries failed for {attempt_label}, using base instruction.")
    return base_def, base_ex


if __name__ == "__main__":
    bot = Chatbot(
        system_prompt="You strictly output valid JSON format.",
        model="gpt-3.5-turbo",
        temperature=1.0,
        max_tokens=500,
    )

    all_results = {}

    for task in TASKS:
        dataset    = task["dataset"]
        task_desc  = task["task_desc"]
        output_file = task["output_file"]

        print("=" * 80)
        print(f"Running Baseline: RanInstruct | Dataset: {dataset} | PLM: {plm}")
        print(f"Generating {NUM_MUTATIONS} random instructions for {task_desc}")
        print("=" * 80)

        base_inst = Instruction(dataset=dataset)
        base_def  = base_inst.definition
        base_ex   = base_inst.example

        print(f"\n[Base Definition]\n{base_def.strip()}")

        results = []

        for i in range(NUM_MUTATIONS):
            print(f"\n{'-'*60}")
            print(f"[{dataset}] Generating instruction {i+1}/{NUM_MUTATIONS} ...")

            new_def, new_ex = generate_random_instruction(
                bot, base_def, base_ex, task_desc,
                attempt_label=f"{dataset} #{i+1}"
            )

            print(f"  Definition: {new_def.strip()[:80]}...")

            inst = Instruction(definition=new_def, example=new_ex, dataset=dataset)
            print(f"  Evaluating ...")

            obj = Objective(inst_individual=inst, dataset=dataset, plm=plm)

            try:
                length            = float(obj.objectives[0])
                perplexity        = float(obj.objectives[1])
                reciprocal_metric = float(obj.objectives[2])
                avg_metric = (1.0 / reciprocal_metric) / 4.0 if reciprocal_metric != 0 else 0
            except Exception:
                length, perplexity, avg_metric = 0.0, 0.0, 0.0

            res = {
                "dataset":     dataset,
                "instruction": new_def,
                "example":     new_ex,
                "objectives":  list(obj.objectives),
                "metrics":     obj.metric,
                "avg_metric":  avg_metric,
            }
            results.append(res)

            print(f"  Length={length:.0f} | Perplexity={perplexity:.2f} | Avg metric={avg_metric*100:.2f}%")

        # Save per-task results
        pickle.dump(results, open(output_file, "wb"))
        print(f"\nSaved {len(results)} results to: {output_file}")

        # Best result for summary (lowest reciprocal_metric = highest performance)
        best = min(results, key=lambda r: r["objectives"][2] if r["objectives"][2] > 0 else float("inf"))
        best_avg = best["avg_metric"]
        print(f"Best avg metric for {dataset}: {best_avg*100:.2f}%")

        all_results[dataset] = results

    # Save combined results
    pickle.dump(all_results, open("raninstruct_all.pkl", "wb"))

    print("\n" + "=" * 80)
    print("All RanInstruct tasks done. Summary (best per dataset):")
    for ds, results in all_results.items():
        best = min(results, key=lambda r: r["objectives"][2] if r["objectives"][2] > 0 else float("inf"))
        accs = [r["metrics"].get("accuracy", 0) * 100 for r in results if isinstance(r["metrics"], dict)]
        print(f"  {ds}: best accuracy={max(accs):.2f}% | best avg_metric={best['avg_metric']*100:.2f}%")
    print("Combined results saved to: raninstruct_all.pkl")
    print("=" * 80)
