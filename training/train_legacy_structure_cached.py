from __future__ import annotations
import argparse,pickle,json,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
PROGRESS=ROOT/"training/legacy_structure_cached_progress.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=0.002)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:items=pickle.load(f)
    hidden_by_index={}
    for shard in range(8):
        rows=torch.load(
            ROOT/f"training/cache/hidden_{shard:02d}_of_08.pt",
            map_location="cpu",weights_only=False
        )
        for row in rows:hidden_by_index[row["index"]]=row["hidden"].float()

    indices=[i for i,x in enumerate(items) if x["category"]=="legacy_anchor"]
    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    for p in model.parameters():p.requires_grad=False
    prefixes=(
        "question_pool_score","projection_head","projection_arity_head",
        "clause_head","direction_head","continuation_head",
    )
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad=True;params.append(p)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    model.eval()
    metrics=[]

    for epoch in range(1,args.epochs+1):
        order=list(indices);random.Random(args.seed+epoch).shuffle(order)
        total=pc=ac=cc=cb=dc=dtc=contc=0
        loss_sum=0.0
        for start in range(0,len(order),48):
            batch=order[start:start+48]
            summaries=[];pt=[];at=[];ct=[];cont=[];ds=[];dtargets=[]
            for index in batch:
                item=items[index];hidden=hidden_by_index[index]
                summary=model.question_summary_from_hidden(hidden,item["schema_spans"]["question"])
                summaries.append(summary);t=item["targets"]
                pt.append(t["projection"]);at.append(t["arity"]);ct.append(t["clauses"]);cont.append(t["continuation"])
                if t["direction"] in (1,2):
                    ds.append(summary);dtargets.append(t["direction"])
            summaries=torch.stack(summaries)
            pt=torch.tensor(pt);at=torch.tensor(at);ct=torch.stack(ct);cont=torch.tensor(cont)
            pl=model.projection_head(summaries);al=model.projection_arity_head(summaries)
            cl=model.clause_head(summaries);conl=model.continuation_head(summaries)
            p_loss=F.cross_entropy(pl,pt)
            a_loss=F.cross_entropy(al,at)
            c_loss=F.binary_cross_entropy_with_logits(cl,ct)
            cont_loss=F.cross_entropy(conl,cont)
            if ds:
                dss=torch.stack(ds);dtt=torch.tensor(dtargets)
                dl=model.direction_head(dss);d_loss=F.cross_entropy(dl,dtt)
                dc+=int((dl.detach().argmax(-1)==dtt).sum());dtc+=len(dtt)
            else:d_loss=p_loss.new_zeros(())
            loss=p_loss+a_loss+2.0*c_loss+1.5*cont_loss+d_loss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

            n=len(batch);total+=n;loss_sum+=float(loss.detach())*n
            pc+=int((pl.detach().argmax(-1)==pt).sum());ac+=int((al.detach().argmax(-1)==at).sum())
            cc+=int(((torch.sigmoid(cl.detach())>=0.5).float()==ct).sum());cb+=n*3
            contc+=int((conl.detach().argmax(-1)==cont).sum())
        m={
            "epoch":epoch,"examples":total,"loss":loss_sum/max(total,1),
            "projection_accuracy":pc/max(total,1),"arity_accuracy":ac/max(total,1),
            "clause_bit_accuracy":cc/max(cb,1),"direction_accuracy":dc/max(dtc,1),
            "continuation_accuracy":contc/max(total,1),
        }
        metrics.append(m);print(json.dumps(m))
    torch.save(model.state_dict(),CHECKPOINT)
    PROGRESS.write_text(json.dumps({"runs":metrics},indent=2),encoding="utf-8")

if __name__=="__main__":main()
