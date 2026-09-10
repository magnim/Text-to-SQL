from __future__ import annotations
import argparse,pickle,json,random,re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
PROGRESS=ROOT/"training/numeric_role_pointer_progress.json"


def question_value_spans(item, tokenizer):
    q_start,q_end=item["schema_spans"]["question"]
    qids=item["input_ids"][q_start:q_end]
    values=re.findall(r"\b\d+(?:\.\d+)?\b",item["question"])
    ordered=[]
    for value in values:
        if value not in ordered:
            ordered.append(value)
    spans=[]
    for value in ordered:
        vids=tokenizer.encode_ids(value)
        occurrences=[]
        for i in range(0,len(qids)-len(vids)+1):
            if qids[i:i+len(vids)]==vids:
                occurrences.append((q_start+i,q_start+i+len(vids)))
        spans.append(occurrences)
    return ordered,spans


def target_value(sql,kind):
    if kind=="where":
        match=re.search(
            r"\bWHERE\s+[A-Za-z_][A-Za-z0-9_]*\s*"
            r"(?:=|>|<)\s*(-?\d+(?:\.\d+)?)",
            sql,re.I,
        )
    else:
        match=re.search(r"\bLIMIT\s+(\d+)",sql,re.I)
    return match.group(1) if match else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--lr",type=float,default=0.003)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:
        items=pickle.load(f)

    hidden={}
    for shard in range(8):
        rows=torch.load(
            ROOT/f"training/cache/hidden_{shard:02d}_of_08.pt",
            map_location="cpu",weights_only=False,
        )
        for row in rows:
            hidden[row["index"]]=row["hidden"].float()

    examples=[]
    for index,item in enumerate(items):
        values,spans=question_value_spans(item,tok)
        if len(values)<2:
            continue
        for kind in ("where","limit"):
            target=target_value(item["sql"],kind)
            if target is None or target not in values:
                continue
            examples.append(
                (index,kind,spans,values.index(target))
            )

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(
        torch.load(CHECKPOINT,map_location="cpu",weights_only=True),
        strict=True,
    )
    for p in model.parameters():
        p.requires_grad=False
    params=[
        model.where_value_score.weight,
        model.limit_value_score.weight,
    ]
    for p in params:
        p.requires_grad=True

    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    model.eval()
    metrics=[]

    for epoch in range(1,args.epochs+1):
        order=list(range(len(examples)))
        random.Random(args.seed+epoch).shuffle(order)
        total=correct=0
        loss_sum=0.0

        for start in range(0,len(order),64):
            batch=[examples[i] for i in order[start:start+64]]
            losses=[]
            batch_correct=0
            for index,kind,spans,target in batch:
                scores=model.score_question_value_spans(
                    hidden[index],spans,kind
                )
                loss=F.cross_entropy(
                    scores.unsqueeze(0),
                    torch.tensor([target],dtype=torch.long),
                )
                losses.append(loss)
                batch_correct+=int(
                    int(scores.detach().argmax())==target
                )
            if not losses:
                continue
            loss=torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step()

            total+=len(losses)
            correct+=batch_correct
            loss_sum+=sum(float(x.detach()) for x in losses)

        m={
            "epoch":epoch,
            "decisions":total,
            "loss":loss_sum/max(total,1),
            "accuracy":correct/max(total,1),
        }
        metrics.append(m)
        print(json.dumps(m))

    torch.save(model.state_dict(),CHECKPOINT)
    PROGRESS.write_text(
        json.dumps({"runs":metrics,"training_decisions":len(examples)},indent=2),
        encoding="utf-8",
    )

if __name__=="__main__":
    main()
