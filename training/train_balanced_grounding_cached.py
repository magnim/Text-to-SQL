from __future__ import annotations
import argparse,pickle,random,json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT

CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
OPTIMIZER=ROOT/"training/balanced_grounding_cached_optimizer.pt"
PROGRESS=ROOT/"training/balanced_grounding_cached_progress.json"
ROLES=["experience","age","price","pay","stock","rating","mileage","amount","quantity"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epoch",type=int,required=True)
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--shards",type=int,default=8)
    ap.add_argument("--target-per-role",type=int,default=55)
    ap.add_argument("--lr",type=float,default=0.001)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:items=pickle.load(f)
    rows=torch.load(
        ROOT/f"training/cache/hidden_{args.shard:02d}_of_{args.shards:02d}.pt",
        map_location="cpu",weights_only=False
    )
    groups={role:[] for role in ROLES}
    for row in rows:
        cat=items[row["index"]]["category"]
        for role in ROLES:
            if cat==f"where_{role}":
                groups[role].append(row)

    rng=random.Random(args.seed+args.epoch*100+args.shard)
    selected=[]
    for role in ROLES:
        group=groups[role]
        if not group:continue
        rng.shuffle(group)
        for i in range(args.target_per_role):
            selected.append(group[i%len(group)])
    rng.shuffle(selected)

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    for p in model.parameters():p.requires_grad=False
    prefixes=(
        "lexical_table_query_projection","lexical_table_key_projection",
        "lexical_column_query_projection","lexical_column_key_projection",
        "where_question_pool_score","where_column_query_projection","where_column_key_projection",
        "clause_head","continuation_head",
    )
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad=True;params.append(p)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    if OPTIMIZER.exists():
        try:opt.load_state_dict(torch.load(OPTIMIZER,map_location="cpu",weights_only=False))
        except Exception:pass
        for g in opt.param_groups:g["lr"]=args.lr

    totals={"loss":0.0,"n":0,"table":0,"tablet":0,"where":0,"wheret":0,"clause":0,"clauset":0,"cont":0}
    batch_size=48
    for start in range(0,len(selected),batch_size):
        batch=selected[start:start+batch_size]
        losses=[];summaries=[];clauses=[]
        table_meta=[];where_meta=[]
        for row in batch:
            item=items[row["index"]];hidden=row["hidden"].float()
            qspan=item["schema_spans"]["question"];t=item["targets"];schema=item["schema"];table=t["table"]
            summary=model.question_summary_from_hidden(hidden,qspan)
            summaries.append(summary);clauses.append(t["clauses"])

            tables=list(schema)
            fields=[[tok.encode_ids(tab)]+[tok.encode_ids(c) for c in schema[tab]] for tab in tables]
            ts=model.score_schema_fields_from_question(hidden,qspan,fields,"table")
            ti=tables.index(table)
            losses.append(2.5*F.cross_entropy(ts.unsqueeze(0),torch.tensor([ti])))
            table_meta.append((int(ts.detach().argmax()),ti))

            candidates=list(schema[table]);tokens=[tok.encode_ids(c) for c in candidates]
            target=t["where_column"]
            if target in candidates:
                ws=model.score_clause_columns_from_question(hidden,qspan,tokens,"where")
                ti=candidates.index(target)
                losses.append(3.0*F.cross_entropy(ws.unsqueeze(0),torch.tensor([ti])))
                where_meta.append((int(ws.detach().argmax()),ti))
                sem=model.score_schema_fields_from_question(hidden,qspan,[[x] for x in tokens],"column")
                losses.append(F.cross_entropy(sem.unsqueeze(0),torch.tensor([ti])))

        summaries=torch.stack(summaries);clauses=torch.stack(clauses)
        cl=model.clause_head(summaries)
        cont=model.continuation_head(summaries)
        clause_loss=F.binary_cross_entropy_with_logits(cl,clauses)
        cont_targets=torch.zeros(len(batch),dtype=torch.long)
        cont_loss=F.cross_entropy(cont,cont_targets)
        ground=torch.stack(losses).mean()
        loss=2.0*ground+1.5*clause_loss+1.0*cont_loss
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

        n=len(batch);totals["loss"]+=float(loss.detach())*n;totals["n"]+=n
        totals["table"]+=sum(int(a==b) for a,b in table_meta);totals["tablet"]+=len(table_meta)
        totals["where"]+=sum(int(a==b) for a,b in where_meta);totals["wheret"]+=len(where_meta)
        totals["clause"]+=int(((torch.sigmoid(cl.detach())>=0.5).float()==clauses).sum());totals["clauset"]+=n*3
        totals["cont"]+=int((cont.detach().argmax(-1)==cont_targets).sum())

    torch.save(model.state_dict(),CHECKPOINT);torch.save(opt.state_dict(),OPTIMIZER)
    metrics={
        "epoch":args.epoch,"shard":args.shard,"examples":totals["n"],
        "loss":totals["loss"]/max(totals["n"],1),
        "table_accuracy":totals["table"]/max(totals["tablet"],1),
        "where_accuracy":totals["where"]/max(totals["wheret"],1),
        "clause_bit_accuracy":totals["clause"]/max(totals["clauset"],1),
        "continuation_accuracy":totals["cont"]/max(totals["n"],1),
        "source_role_counts":{r:len(groups[r]) for r in ROLES},
    }
    progress={"runs":[]}
    if PROGRESS.exists():progress=json.loads(PROGRESS.read_text(encoding="utf-8"))
    progress.setdefault("runs",[]).append(metrics)
    PROGRESS.write_text(json.dumps(progress,indent=2),encoding="utf-8")
    print(json.dumps(metrics))

if __name__=="__main__":main()
