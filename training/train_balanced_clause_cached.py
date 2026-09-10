from __future__ import annotations
import argparse,pickle,json,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
PROGRESS=ROOT/"training/balanced_clause_cached_progress.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8)
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

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(
        torch.load(CHECKPOINT,map_location="cpu",weights_only=True),
        strict=True
    )
    for p in model.parameters():
        p.requires_grad=False
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(("clause_head","continuation_head")):
            p.requires_grad=True
            params.append(p)

    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    model.eval()

    clause_targets=torch.stack([x["targets"]["clauses"] for x in items])
    positives=clause_targets.sum(0)
    pos_weight=(len(items)-positives)/positives.clamp_min(1.0)

    cont_counts=torch.zeros(4)
    for x in items:
        cont_counts[x["targets"]["continuation"]]+=1
    cont_weights=cont_counts.sum()/(4*cont_counts.clamp_min(1.0))

    metrics=[]
    base_indices=list(range(len(items)))

    for epoch in range(1,args.epochs+1):
        order=list(base_indices)
        random.Random(args.seed+epoch).shuffle(order)
        total=0
        loss_sum=0.0
        clause_correct=0
        clause_bits=0
        cont_correct=0

        for start in range(0,len(order),64):
            batch=order[start:start+64]
            summaries=[]
            ctargets=[]
            conttargets=[]
            for index in batch:
                item=items[index]
                hidden=hidden_by_index[index]
                summaries.append(
                    model.question_summary_from_hidden(
                        hidden,item["schema_spans"]["question"]
                    )
                )
                ctargets.append(item["targets"]["clauses"])
                conttargets.append(item["targets"]["continuation"])

            summaries=torch.stack(summaries)
            ctargets=torch.stack(ctargets)
            conttargets=torch.tensor(conttargets,dtype=torch.long)

            clogits=model.clause_head(summaries)
            contlogits=model.continuation_head(summaries)
            c_loss=F.binary_cross_entropy_with_logits(
                clogits,ctargets,pos_weight=pos_weight
            )
            cont_loss=F.cross_entropy(
                contlogits,conttargets,weight=cont_weights
            )
            loss=1.5*c_loss+cont_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step()

            n=len(batch)
            total+=n
            loss_sum+=float(loss.detach())*n
            clause_correct+=int(
                ((torch.sigmoid(clogits.detach())>=0.5).float()==ctargets).sum()
            )
            clause_bits+=n*3
            cont_correct+=int(
                (contlogits.detach().argmax(-1)==conttargets).sum()
            )

        m={
            "epoch":epoch,
            "examples":total,
            "loss":loss_sum/max(total,1),
            "clause_bit_accuracy":clause_correct/max(clause_bits,1),
            "continuation_accuracy":cont_correct/max(total,1),
        }
        metrics.append(m)
        print(json.dumps(m))

    torch.save(model.state_dict(),CHECKPOINT)
    PROGRESS.write_text(json.dumps({"runs":metrics},indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
