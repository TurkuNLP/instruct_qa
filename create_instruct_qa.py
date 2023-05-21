import jsonlines
from datasets import load_dataset

squad = load_dataset("TurkuNLP/squad_v2_fi")

counter = 0
for line in squad["train"]:
    context = line["context"]
    instruction = line["question"]
    response = " ".join(str(x) for x in line["answers"]["text"])
    id = f"squad_v2_train_{counter}"

    if len(response) <= 0:
        instruction += ' Jos vastausta ei löydy, vastaa "Ei vastausta"'
        response = "Ei vastausta"

    line = {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": "closed_qa",
        "id": id
    }

    counter += 1

    with jsonlines.open('instruct_qa_fi.jsonl', mode='a') as writer:
        writer.write(line)

counter = 0
for line in squad["validation"]:
    context = line["context"]
    instruction = line["question"]
    response = " ".join(str(x) for x in line["answers"]["text"])
    id = f"squad_v2_validation_{counter}"

    if len(response) <= 0:
        instruction += ' Jos vastausta ei löydy, vastaa "Ei vastausta"'
        response = "Ei vastausta"

    line = {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": "closed_qa",
        "id": id
    }

    counter += 1

    with jsonlines.open('instruct_qa_fi.jsonl', mode='a') as writer:
        writer.write(line)


nq = load_dataset("nq_fi.py")

counter = 0
for line in nq["train"]:
    context = line["context"]
    instruction = line["question"]
    response = " ".join(str(x) for x in line["answers"]["text"])
    id = f"nq_train_{counter}"

    if len(response) <= 0:
        instruction += ' Jos vastausta ei löydy, vastaa "Ei vastausta"'
        response = "Ei vastausta"

    line = {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": "closed_qa",
        "id": id
    }

    counter += 1

    with jsonlines.open('instruct_qa_fi.jsonl', mode='a') as writer:
        writer.write(line)

counter = 0
for line in nq["validation"]:
    context = line["context"]
    instruction = line["question"]
    response = " ".join(str(x) for x in line["answers"]["text"])
    id = f"nq_validation_{counter}"

    if len(response) <= 0:
        instruction += ' Jos vastausta ei löydy, vastaa "Ei vastausta"'
        response = "Ei vastausta"

    line = {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": "closed_qa",
        "id": id
    }

    counter += 1

    with jsonlines.open('instruct_qa_fi.jsonl', mode='a') as writer:
        writer.write(line)
