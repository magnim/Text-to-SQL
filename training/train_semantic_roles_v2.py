from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2,identifier_to_text
from training.semantic_pretraining_v2 import CONCEPTS
CHECKPOINT=ROOT/"semantic_schema_encoder_v2.pt"
WHERE_TEMPLATES=["show {d} for records where {c}","list {d}; keep entries that are {c}","return {d} from records {c}","show records {c} and include {d}","give me {d} for items that are {c}","{d} for entries {c}"]
ORDER_TEMPLATES=["show {d} ordered by {t}","sort records by {t} and include {d}","list {d}, ranked using {t}","arrange entries by {t}; show {d}","order the records on {t} and return {d}"]

def build(seed=42,per=48):
    rng=random.Random(seed);concepts=sorted(CONCEPTS);out={"where":[],"order":[]}
    for tc in concepts:
        target=CONCEPTS[tc];others=[c for c in concepts if c!=tc]
        for _ in range(per):
            dc=rng.choice(others);d=identifier_to_text(rng.choice(CONCEPTS[dc]["fields"]));tf=rng.choice(target["fields"])
            cond=rng.choice(target["questions"]);out["where"].append({"concept":tc,"question":rng.choice(WHERE_TEMPLATES).format(d=d,c=cond),"focus":cond,"field":tf})
            tl=identifier_to_text(rng.choice(target["fields"]));out["order"].append({"concept":tc,"question":rng.choice(ORDER_TEMPLATES).format(d=d,t=tl),"focus":tl,"field":tf})
        for q in target["questions"]:out["where"].append({"concept":tc,"question":q,"focus":q,"field":rng.choice(target["fields"])})
    return out

def batch_unique(examples,rng):
    groups=defaultdict(list)
    for x in examples:groups[x["concept"]].append(x)
    return [rng.choice(groups[c]) for c in sorted(groups)]

def encode_focus(tok,qs,focuses,maxlen,pad):
    enc=[];fm=[]
    for q,f in zip(qs,focuses):
        ids=tok.encode_ids(q)[:maxlen];fid=tok.encode_ids(f);mask=[0.]*len(ids)
        for s in range(max(0,len(ids)-len(fid)+1)):
            if fid and ids[s:s+len(fid)]==fid:
                for j in range(s,s+len(fid)):mask[j]=1.
                break
        if not any(mask):mask=[1.]*len(ids)
        enc.append(ids);fm.append(mask)
    ml=max(max(len(x),1) for x in enc);rows=[];masks=[];frows=[]
    for ids,m in zip(enc,fm):
        ids=ids or [pad];m=m or [1.];n=len(ids);rows.append(ids+[pad]*(ml-n));masks.append([1]*n+[0]*(ml-n));frows.append(m+[0.]*(ml-len(m)))
    return torch.tensor(rows),torch.tensor(masks,dtype=torch.bool),torch.tensor(frows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=500);ap.add_argument("--lr",type=float,default=.001);ap.add_argument("--seed",type=int,default=101)
    a=ap.parse_args();torch.manual_seed(a.seed);rng=random.Random(a.seed)
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"));pad=tok.vocab["<PAD>"]
    m=SchemaSemanticEncoderV2(len(tok.vocab),64,64,4,160,2,pad)
    m.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    for p in m.parameters():p.requires_grad=False
    params=[]
    for n,p in m.named_parameters():
        if n.startswith(("role_token_classifier.","role_projection.")):p.requires_grad=True;params.append(p)
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01);examples=build(a.seed);m.eval();history=[]
    for step in range(1,a.steps+1):
        loss_sum=0.;corr=tot=0
        for role in ("where","order"):
            batch=batch_unique(examples[role],rng);qs=[x["question"] for x in batch];fs=[x["focus"] for x in batch]
            qids,qmask,fmask=encode_focus(tok,qs,fs,64,pad);tl,states,mask=m.role_token_logits(qids,qmask,role)
            w=torch.softmax(tl,dim=-1);pooled=(states*w.unsqueeze(-1)).sum(1);q=F.normalize(m.role_projection[role](pooled),dim=-1)
            with torch.no_grad():f=m.encode_texts(tok,[identifier_to_text(x["field"]) for x in batch])
            temp=m.log_temperature.exp().detach().clamp(.02,.30);logits=(q@f.T)/temp;labels=torch.arange(len(batch))
            retr=F.cross_entropy(logits,labels);vl=tl[mask];vt=fmask[mask];pos=vt.sum().clamp_min(1.);neg=(1-vt).sum().clamp_min(1.)
            token=F.binary_cross_entropy_with_logits(vl,vt,pos_weight=(neg/pos).detach());att=-torch.log((w*fmask).sum(-1).clamp_min(1e-8)).mean()
            loss=retr+1.5*token+.5*att;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            loss_sum+=float(loss.detach());corr+=int((logits.detach().argmax(-1)==labels).sum());tot+=len(batch)
        if step%50==0 or step==a.steps:
            item={"step":step,"loss":loss_sum/2,"retrieval_accuracy":corr/max(tot,1)};history.append(item);print(json.dumps(item))
    torch.save(m.state_dict(),CHECKPOINT);(ROOT/"training/semantic_roles_v2_metrics.json").write_text(json.dumps({"history":history},indent=2))
if __name__=="__main__":main()
