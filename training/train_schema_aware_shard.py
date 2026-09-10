from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from pytorch_impl.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from training.schema_conditioned_dataset import build_schema_conditioned_examples
from training.schema_aware_training import (
    encode_example,batchify,load_legacy_initialization,trainable_aux_parameters
)

LEGACY=ROOT/"tiny_gpt_text_to_sql_with_in_v4.pt"
CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
OPTIMIZER=ROOT/"training/schema_aware_final_optimizer.pt"
PROGRESS=ROOT/"training/schema_aware_shard_progress.json"

def oversampled(items,seed):
    out=[]
    for item in items:
        out.append(item)
        cat=item["category"]
        if any(k in cat for k in ("where_order","order_limit","multi_projection","where_mileage")):
            out.extend([item,item])
        if cat=="legacy_anchor":out.append(item)
    random.Random(seed).shuffle(out)
    return out

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
    ap.add_argument("--batch-size",type=int,default=56)
    args=ap.parse_args()

    random.seed(args.seed);torch.manual_seed(args.seed)
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    raw=build_schema_conditioned_examples(args.seed)
    encoded=[encode_example(e,tok) for e in raw]
    effective=oversampled(encoded,args.seed+args.epoch)
    subset=effective[args.shard::args.shards]

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    else:
        load_legacy_initialization(model,LEGACY)
    params=trainable_aux_parameters(model)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    if OPTIMIZER.exists() and CHECKPOINT.exists():
        opt.load_state_dict(torch.load(OPTIMIZER,map_location="cpu",weights_only=False))
        for group in opt.param_groups:group["lr"]=args.lr

    pweights=weights(encoded,"projection",5)
    aweights=weights(encoded,"arity",3)
    cpos=(len(encoded)-torch.stack([x["targets"]["clauses"] for x in encoded]).sum(0))/torch.stack([x["targets"]["clauses"] for x in encoded]).sum(0).clamp_min(1.0)
    contweights=weights(encoded,"continuation",4)
    pad=tok.vocab["<PAD>"]
    totals={k:0.0 for k in ["loss","ground_loss"]}
    counts={k:0 for k in [
        "examples","projection_correct","arity_correct","clause_correct","clause_bits",
        "direction_correct","direction_total","continuation_correct",
        "table_correct","table_total","where_correct","where_total","order_correct","order_total",
        "normalized_correct","normalized_total","context_correct","context_total"
    ]}
    model.eval()

    for start in range(0,len(subset),args.batch_size):
        batch=subset[start:start+args.batch_size]
        ids,roles=batchify(batch,pad)
        with torch.no_grad():
            hidden=model.get_hidden_states(ids,roles).detach()

        summaries=[];pt=[];at=[];ct=[];cont=[];dir_s=[];dir_t=[]
        ground=[];metas={k:[] for k in ("table","where","order","normalized","context")}
        for bi,item in enumerate(batch):
            qspan=item["schema_spans"]["question"]
            summary=model.question_summary_from_hidden(hidden[bi],qspan)
            summaries.append(summary);t=item["targets"]
            pt.append(t["projection"]);at.append(t["arity"]);ct.append(t["clauses"]);cont.append(t["continuation"])
            if t["direction"] in (1,2):dir_s.append(summary);dir_t.append(t["direction"])

            schema=item["schema"];table=t["table"]
            if table in schema:
                tables=list(schema)
                field_groups=[
                    [tok.encode_ids(tab)]+[tok.encode_ids(c) for c in schema[tab]]
                    for tab in tables
                ]
                ts=model.score_schema_fields_from_question(hidden[bi],qspan,field_groups,"table")
                ti=tables.index(table)
                ground.append(2.0*F.cross_entropy(ts.unsqueeze(0),torch.tensor([ti])))
                metas["table"].append((int(ts.detach().argmax()),ti))

                candidates=list(schema[table]);token_lists=[tok.encode_ids(c) for c in candidates]
                semantic_targets=[]
                for c in t["projection_columns"]+[t["where_column"],t["order_column"]]:
                    if c in candidates and c not in semantic_targets:semantic_targets.append(c)
                for c in semantic_targets:
                    sem=model.score_schema_fields_from_question(hidden[bi],qspan,[[x] for x in token_lists],"column")
                    ground.append(0.4*F.cross_entropy(sem.unsqueeze(0),torch.tensor([candidates.index(c)])))

                if t["where_column"] in candidates:
                    ws=model.score_clause_columns_from_question(hidden[bi],qspan,token_lists,"where")
                    ti=candidates.index(t["where_column"])
                    ground.append(3.0*F.cross_entropy(ws.unsqueeze(0),torch.tensor([ti])))
                    metas["where"].append((int(ws.detach().argmax()),ti))
                if t["order_column"] in candidates:
                    os=model.score_clause_columns_from_question(hidden[bi],qspan,token_lists,"order")
                    ti=candidates.index(t["order_column"])
                    ground.append(3.0*F.cross_entropy(os.unsqueeze(0),torch.tensor([ti])))
                    metas["order"].append((int(os.detach().argmax()),ti))

                target_col=t["where_column"] or (t["projection_columns"][0] if t["projection_columns"] else None)
                lower=[tok.encode_ids(c.lower()) for c in candidates]
                if target_col in candidates and all(lower):
                    ns=model.score_normalized_columns_from_question(hidden[bi],qspan,lower)
                    ti=candidates.index(target_col)
                    ground.append(0.8*F.cross_entropy(ns.unsqueeze(0),torch.tensor([ti])))
                    metas["normalized"].append((int(ns.detach().argmax()),ti))

            for d in item["contextual"]:
                pos=d["position"]
                if pos<0 or pos>=hidden.shape[1]:continue
                scores=model.score_schema_spans_from_hidden(hidden[bi],pos,d["candidate_spans"],d["kind"])
                ti=d["target_index"]
                ground.append(0.8*F.cross_entropy(scores.unsqueeze(0),torch.tensor([ti])))
                metas["context"].append((int(scores.detach().argmax()),ti))

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
            counts["direction_correct"]+=int((dl.detach().argmax(-1)==dt).sum());counts["direction_total"]+=len(dt)
        else:d_loss=p_loss.new_zeros(())
        g_loss=torch.stack(ground).mean() if ground else p_loss.new_zeros(())
        loss=p_loss+a_loss+c_loss+1.5*cont_loss+0.8*d_loss+2.0*g_loss

        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

        n=len(batch);counts["examples"]+=n;totals["loss"]+=float(loss.detach())*n
        totals["ground_loss"]+=sum(float(x.detach()) for x in ground)
        counts["projection_correct"]+=int((pl.detach().argmax(-1)==pt).sum())
        counts["arity_correct"]+=int((al.detach().argmax(-1)==at).sum())
        counts["clause_correct"]+=int(((torch.sigmoid(cl.detach())>=0.5).float()==ct).sum());counts["clause_bits"]+=n*3
        counts["continuation_correct"]+=int((conl.detach().argmax(-1)==cont).sum())
        for name,meta in metas.items():
            counts[name+"_correct"]+=sum(int(a==b) for a,b in meta);counts[name+"_total"]+=len(meta)

    torch.save(model.state_dict(),CHECKPOINT);torch.save(opt.state_dict(),OPTIMIZER)
    n=max(counts["examples"],1)
    metrics={
        "epoch":args.epoch,"shard":args.shard,"examples":counts["examples"],
        "source_examples":len(encoded),"effective_examples":len(effective),
        "mean_loss":totals["loss"]/n,
        "projection_accuracy":counts["projection_correct"]/n,
        "arity_accuracy":counts["arity_correct"]/n,
        "clause_bit_accuracy":counts["clause_correct"]/max(counts["clause_bits"],1),
        "direction_accuracy":counts["direction_correct"]/max(counts["direction_total"],1),
        "continuation_accuracy":counts["continuation_correct"]/n,
        "table_accuracy":counts["table_correct"]/max(counts["table_total"],1),
        "where_column_accuracy":counts["where_correct"]/max(counts["where_total"],1),
        "order_column_accuracy":counts["order_correct"]/max(counts["order_total"],1),
        "normalized_accuracy":counts["normalized_correct"]/max(counts["normalized_total"],1),
        "contextual_accuracy":counts["context_correct"]/max(counts["context_total"],1),
        "learning_rate":args.lr,
    }
    progress={"runs":[]}
    if PROGRESS.exists():progress=json.loads(PROGRESS.read_text(encoding="utf-8"))
    progress.setdefault("runs",[]).append(metrics)
    PROGRESS.write_text(json.dumps(progress,indent=2),encoding="utf-8")
    print(json.dumps(metrics))

if __name__=="__main__":main()
