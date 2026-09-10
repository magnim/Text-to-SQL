from __future__ import annotations
import argparse,json,pickle,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from training.schema_aware_training import trainable_aux_parameters

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
OPTIMIZER=ROOT/"training/schema_aware_cached_optimizer.pt"
PROGRESS=ROOT/"training/schema_aware_cached_progress.json"

def weights(items,key,n):
    counts=torch.zeros(n)
    for x in items:counts[x["targets"][key]]+=1
    return counts.sum()/(n*counts.clamp_min(1.0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epoch",type=int,required=True)
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--shards",type=int,default=8)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--lr",type=float,default=0.0015)
    ap.add_argument("--batch-size",type=int,default=32)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:items=pickle.load(f)
    rows=torch.load(
        ROOT/f"training/cache/hidden_{args.shard:02d}_of_{args.shards:02d}.pt",
        map_location="cpu",weights_only=False
    )

    expanded=[]
    for row in rows:
        item=items[row["index"]]
        expanded.append(row)
        cat=item["category"]
        if any(k in cat for k in ("where_order","order_limit","multi_projection","where_mileage")):
            expanded.extend([row,row])
        if cat=="legacy_anchor":expanded.append(row)
    random.Random(args.seed+args.epoch*100+args.shard).shuffle(expanded)

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    params=trainable_aux_parameters(model)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    if OPTIMIZER.exists():
        try:opt.load_state_dict(torch.load(OPTIMIZER,map_location="cpu",weights_only=False))
        except Exception:pass
        for g in opt.param_groups:g["lr"]=args.lr

    pweights=weights(items,"projection",5)
    aweights=weights(items,"arity",3)
    clause_stack=torch.stack([x["targets"]["clauses"] for x in items])
    cpos=(len(items)-clause_stack.sum(0))/clause_stack.sum(0).clamp_min(1.0)
    contweights=weights(items,"continuation",4)

    counts={k:0 for k in [
        "examples","p","a","cb","cbt","d","dt","cont",
        "table","tablet","where","wheret","order","ordert","norm","normt","ctx","ctxt"
    ]}
    total_loss=0.0
    model.eval()

    for start in range(0,len(expanded),args.batch_size):
        batch=expanded[start:start+args.batch_size]
        summaries=[];pt=[];at=[];ct=[];cont=[];dir_s=[];dir_t=[]
        ground=[];meta={k:[] for k in ("table","where","order","norm","ctx")}

        for row in batch:
            item=items[row["index"]]
            hidden=row["hidden"].float()
            qspan=item["schema_spans"]["question"]
            summary=model.question_summary_from_hidden(hidden,qspan)
            summaries.append(summary)
            t=item["targets"]
            pt.append(t["projection"]);at.append(t["arity"]);ct.append(t["clauses"]);cont.append(t["continuation"])
            if t["direction"] in (1,2):dir_s.append(summary);dir_t.append(t["direction"])

            schema=item["schema"];table=t["table"]
            if table in schema:
                tables=list(schema)
                field_groups=[
                    [tok.encode_ids(tab)]+[tok.encode_ids(c) for c in schema[tab]]
                    for tab in tables
                ]
                ts=model.score_schema_fields_from_question(hidden,qspan,field_groups,"table")
                ti=tables.index(table)
                ground.append(2.0*F.cross_entropy(ts.unsqueeze(0),torch.tensor([ti])))
                meta["table"].append((int(ts.detach().argmax()),ti))

                candidates=list(schema[table]);token_lists=[tok.encode_ids(c) for c in candidates]
                semantic_targets=[]
                for c in t["projection_columns"]+[t["where_column"],t["order_column"]]:
                    if c in candidates and c not in semantic_targets:semantic_targets.append(c)
                for c in semantic_targets:
                    sem=model.score_schema_fields_from_question(hidden,qspan,[[x] for x in token_lists],"column")
                    ground.append(0.35*F.cross_entropy(sem.unsqueeze(0),torch.tensor([candidates.index(c)])))

                if t["where_column"] in candidates:
                    ws=model.score_clause_columns_from_question(hidden,qspan,token_lists,"where")
                    ti=candidates.index(t["where_column"])
                    ground.append(3.0*F.cross_entropy(ws.unsqueeze(0),torch.tensor([ti])))
                    meta["where"].append((int(ws.detach().argmax()),ti))
                if t["order_column"] in candidates:
                    os=model.score_clause_columns_from_question(hidden,qspan,token_lists,"order")
                    ti=candidates.index(t["order_column"])
                    ground.append(3.0*F.cross_entropy(os.unsqueeze(0),torch.tensor([ti])))
                    meta["order"].append((int(os.detach().argmax()),ti))

                target_col=t["where_column"] or (t["projection_columns"][0] if t["projection_columns"] else None)
                lower=[tok.encode_ids(c.lower()) for c in candidates]
                if target_col in candidates and all(lower):
                    ns=model.score_normalized_columns_from_question(hidden,qspan,lower)
                    ti=candidates.index(target_col)
                    ground.append(0.8*F.cross_entropy(ns.unsqueeze(0),torch.tensor([ti])))
                    meta["norm"].append((int(ns.detach().argmax()),ti))

            for d in item["contextual"]:
                pos=d["position"]
                if pos<0 or pos>=hidden.shape[0]:continue
                scores=model.score_schema_spans_from_hidden(hidden,pos,d["candidate_spans"],d["kind"])
                ti=d["target_index"]
                ground.append(0.8*F.cross_entropy(scores.unsqueeze(0),torch.tensor([ti])))
                meta["ctx"].append((int(scores.detach().argmax()),ti))

        summaries=torch.stack(summaries)
        pt=torch.tensor(pt);at=torch.tensor(at);ct=torch.stack(ct);cont=torch.tensor(cont)
        pl=model.projection_head(summaries);al=model.projection_arity_head(summaries)
        cl=model.clause_head(summaries);conl=model.continuation_head(summaries)
        p_loss=F.cross_entropy(pl,pt,weight=pweights)
        a_loss=F.cross_entropy(al,at,weight=aweights)
        c_loss=F.binary_cross_entropy_with_logits(cl,ct,pos_weight=cpos)
        cont_loss=F.cross_entropy(conl,cont,weight=contweights)
        if dir_s:
            ds=torch.stack(dir_s);dt=torch.tensor(dir_t)
            dl=model.direction_head(ds);d_loss=F.cross_entropy(dl,dt)
            counts["d"]+=int((dl.detach().argmax(-1)==dt).sum());counts["dt"]+=len(dt)
        else:d_loss=p_loss.new_zeros(())
        g_loss=torch.stack(ground).mean() if ground else p_loss.new_zeros(())
        loss=p_loss+a_loss+c_loss+1.5*cont_loss+0.8*d_loss+2.0*g_loss
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

        n=len(batch);counts["examples"]+=n;total_loss+=float(loss.detach())*n
        counts["p"]+=int((pl.detach().argmax(-1)==pt).sum())
        counts["a"]+=int((al.detach().argmax(-1)==at).sum())
        counts["cb"]+=int(((torch.sigmoid(cl.detach())>=0.5).float()==ct).sum());counts["cbt"]+=n*3
        counts["cont"]+=int((conl.detach().argmax(-1)==cont).sum())
        for name,m in meta.items():
            counts[name]+=sum(int(x==y) for x,y in m);counts[name+"t"]+=len(m)

    torch.save(model.state_dict(),CHECKPOINT);torch.save(opt.state_dict(),OPTIMIZER)
    n=max(counts["examples"],1)
    metrics={
        "epoch":args.epoch,"shard":args.shard,"examples":counts["examples"],
        "mean_loss":total_loss/n,
        "projection_accuracy":counts["p"]/n,
        "arity_accuracy":counts["a"]/n,
        "clause_bit_accuracy":counts["cb"]/max(counts["cbt"],1),
        "direction_accuracy":counts["d"]/max(counts["dt"],1),
        "continuation_accuracy":counts["cont"]/n,
        "table_accuracy":counts["table"]/max(counts["tablet"],1),
        "where_column_accuracy":counts["where"]/max(counts["wheret"],1),
        "order_column_accuracy":counts["order"]/max(counts["ordert"],1),
        "normalized_accuracy":counts["norm"]/max(counts["normt"],1),
        "contextual_accuracy":counts["ctx"]/max(counts["ctxt"],1),
        "learning_rate":args.lr,
    }
    progress={"runs":[]}
    if PROGRESS.exists():progress=json.loads(PROGRESS.read_text(encoding="utf-8"))
    progress.setdefault("runs",[]).append(metrics)
    PROGRESS.write_text(json.dumps(progress,indent=2),encoding="utf-8")
    print(json.dumps(metrics))

if __name__=="__main__":main()
