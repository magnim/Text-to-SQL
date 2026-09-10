from __future__ import annotations
import argparse,pickle,json,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
PROGRESS=ROOT/"training/projection_pointer_cached_progress.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--lr",type=float,default=0.0015)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:
        items=pickle.load(f)

    hidden_by_index={}
    for shard in range(8):
        rows=torch.load(
            ROOT/f"training/cache/hidden_{shard:02d}_of_08.pt",
            map_location="cpu",weights_only=False
        )
        for row in rows:
            hidden_by_index[row["index"]]=row["hidden"].float()

    indices=[
        i for i,item in enumerate(items)
        if item["category"]=="legacy_anchor"
        and len(item["targets"]["projection_columns"])>=1
    ]

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(
        torch.load(CHECKPOINT,map_location="cpu",weights_only=True),
        strict=True
    )
    for p in model.parameters():
        p.requires_grad=False
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(("column_query_projection","column_key_projection")):
            p.requires_grad=True
            params.append(p)

    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    model.eval()
    metrics=[]

    for epoch in range(1,args.epochs+1):
        order=list(indices)
        random.Random(args.seed+epoch).shuffle(order)
        total_loss=0.0
        total=0
        correct=0

        for start in range(0,len(order),48):
            batch=order[start:start+48]
            losses=[]
            meta=[]

            for index in batch:
                item=items[index]
                hidden=hidden_by_index[index]
                projection_count=len(item["targets"]["projection_columns"])
                column_decisions=[
                    d for d in item["contextual"]
                    if d["kind"]=="column"
                ][:projection_count]

                for decision in column_decisions:
                    pos=decision["position"]
                    if pos<0 or pos>=hidden.shape[0]:
                        continue
                    scores=model.score_schema_spans_from_hidden(
                        hidden,
                        pos,
                        decision["candidate_spans"],
                        "column",
                    )
                    target=decision["target_index"]
                    losses.append(
                        F.cross_entropy(
                            scores.unsqueeze(0),
                            torch.tensor([target],dtype=torch.long),
                        )
                    )
                    meta.append((int(scores.detach().argmax()),target))

            if not losses:
                continue
            loss=torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step()

            total_loss+=sum(float(v.detach()) for v in losses)
            total+=len(losses)
            correct+=sum(int(a==b) for a,b in meta)

        m={
            "epoch":epoch,
            "decisions":total,
            "loss":total_loss/max(total,1),
            "accuracy":correct/max(total,1),
        }
        metrics.append(m)
        print(json.dumps(m))

    torch.save(model.state_dict(),CHECKPOINT)
    PROGRESS.write_text(json.dumps({"runs":metrics},indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
