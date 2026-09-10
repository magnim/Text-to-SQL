from __future__ import annotations
import argparse,pickle,json,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
PROGRESS=ROOT/"training/semantic_structure_cached_progress.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--lr",type=float,default=0.002)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:
        items=pickle.load(f)

    hidden={}
    for shard in range(8):
        rows=torch.load(
            ROOT/f"training/cache/hidden_{shard:02d}_of_08.pt",
            map_location="cpu",weights_only=False
        )
        for row in rows:hidden[row["index"]]=row["hidden"].float()

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(
        torch.load(CHECKPOINT,map_location="cpu",weights_only=True),
        strict=True
    )
    for p in model.parameters():p.requires_grad=False
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(("semantic_clause_head","semantic_continuation_head")):
            p.requires_grad=True;params.append(p)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    model.eval()

    ctargets=torch.stack([x["targets"]["clauses"] for x in items])
    positives=ctargets.sum(0)
    pos_weight=(len(items)-positives)/positives.clamp_min(1.0)
    cont_counts=torch.zeros(4)
    for x in items:cont_counts[x["targets"]["continuation"]]+=1
    cont_weights=cont_counts.sum()/(4*cont_counts.clamp_min(1.0))

    metrics=[]
    base=list(range(len(items)))
    for epoch in range(1,args.epochs+1):
        order=list(base);random.Random(args.seed+epoch).shuffle(order)
        total=0;loss_sum=0.0;cc=cb=contc=0
        for start in range(0,len(order),64):
            batch=order[start:start+64]
            summaries=[];cts=[];conts=[]
            for index in batch:
                item=items[index]
                summaries.append(
                    model.mean_question_state(
                        hidden[index],item["schema_spans"]["question"]
                    )
                )
                cts.append(item["targets"]["clauses"])
                conts.append(item["targets"]["continuation"])
            summaries=torch.stack(summaries)
            cts=torch.stack(cts)
            conts=torch.tensor(conts,dtype=torch.long)
            cl=model.semantic_clause_head(summaries)
            col=model.semantic_continuation_head(summaries)
            c_loss=F.binary_cross_entropy_with_logits(cl,cts,pos_weight=pos_weight)
            cont_loss=F.cross_entropy(col,conts,weight=cont_weights)
            loss=1.5*c_loss+cont_loss
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

            n=len(batch);total+=n;loss_sum+=float(loss.detach())*n
            cc+=int(((torch.sigmoid(cl.detach())>=0.5).float()==cts).sum());cb+=n*3
            contc+=int((col.detach().argmax(-1)==conts).sum())
        m={
            "epoch":epoch,"examples":total,"loss":loss_sum/max(total,1),
            "clause_bit_accuracy":cc/max(cb,1),
            "continuation_accuracy":contc/max(total,1),
        }
        metrics.append(m);print(json.dumps(m))
    torch.save(model.state_dict(),CHECKPOINT)
    PROGRESS.write_text(json.dumps({"runs":metrics},indent=2),encoding="utf-8")

if __name__=="__main__":main()
