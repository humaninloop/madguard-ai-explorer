import json
from pathlib import Path

# Load SQuAD v1.1 data
with open("archive/train-v1.1.json", "r") as f:
    squad_data = json.load(f)

qa_pairs = []
max_pairs = 20

for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            question = qa["question"]
            answers = qa.get("answers", [])
            if answers:
                answer = answers[0]["text"]
                qa_pairs.append(f"Q: {question}\nA: {answer}\n")
            if len(qa_pairs) >= max_pairs:
                break
        if len(qa_pairs) >= max_pairs:
            break
    if len(qa_pairs) >= max_pairs:
        break

# Save to training_set.txt
Path("training_set.txt").write_text("\n".join(qa_pairs))
print("✅ Wrote real data to training_set.txt")
