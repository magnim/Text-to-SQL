from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

from pytorch_impl.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from training.schema_conditioned_dataset import build_schema_conditioned_examples
from training.schema_aware_training import (
    encode_example,batchify,load_legacy_initialization,trainable_aux_parameters,sql_target_loss
)

LEGACY=ROOT/"tiny_gpt_text_to_sql_with_in_v4.pt"
CHECKPOINT=ROOT/"tiny_gpt_schema_aware_final.pt"
OPTIMIZER=ROOT/"training/schema_aware_final_optimizer.pt"
PROGRESS=ROOT/"training/schema_aware_final_progress.json"


def oversampled(items,seed):
    expanded=[]
    for item in items:
        expanded.append(item)
        cat=item["category"]
        if any(key in cat for key in ("where_order","order_limit","multi_projection","where_mileage")):
            expanded.extend([item,item])
        if cat in {"legacy_anchor"}:
            expanded.append(item)
    random.Random(seed).shuffle(expanded)
    return expanded


def class_weights(items,key,classes):
    counts=torch.zeros(classes,dtype=torch.float32)
    for item in items:counts[item["targets"][key]]+=1
    return counts,counts.sum()/(classes*counts.clamp_min(1.0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--lr",type=float,default=0.0015)
    ap.add_argument("--batch-size",type=int,default=56)
    args=ap.parse_args()

    random.seed(args.seed);torch.manual_seed(args.seed)
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    raw=build_schema_conditioned_examples(args.seed)
    encoded=[encode_example(e,tok) for e in raw]

    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    else:
        load_legacy_initialization(model,LEGACY)

    params=trainable_aux_parameters(model)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    if OPTIMIZER.exists() and CHECKPOINT.exists():
        try:opt.load_state_dict(torch.load(OPTIMIZER,map_location="cpu",weights_only=False))
        except Exception:pass

    pad=tok.vocab["<PAD>"]
    pcounts,pweights=class_weights(encoded,"projection",5)
    acounts,aweights=class_weights(encoded,"arity",3)
    ccounts=torch.stack([x["targets"]["clauses"] for x in encoded]).sum(0)
    cpos=(len(encoded)-ccounts)/ccounts.clamp_min(1.0)
    contcounts,contweights=class_weights(encoded,"continuation",4)

    all_metrics=[]
    for epoch in range(1,args.epochs+1):
        effective=oversampled(encoded,args.seed+epoch)
        totals={
            "examples":0,"loss":0.0,"projection_correct":0,"arity_correct":0,
            "clause_correct":0,"clause_bits":0,"direction_correct":0,"direction_total":0,
            "continuation_correct":0,"table_correct":0,"table_total":0,
            "where_correct":0,"where_total":0,"order_correct":0,"order_total":0,
            "normalized_correct":0,"normalized_total":0,"context_correct":0,"context_total":0,
            "ground_loss":0.0,
        }
        model.eval()
        for start in range(0,len(effective),args.batch_size):
            batch=effective[start:start+args.batch_size]
            ids,roles=batchify(batch,pad)
            with torch.no_grad():
                hidden=model.get_hidden_states(ids,roles).detach()

            summaries=[];pt=[];at=[];ct=[];cont=[];dir_s=[];dir_t=[]
            losses_ground=[];table_meta=[];where_meta=[];order_meta=[];norm_meta=[];ctx_meta=[]

            for bi,item in enumerate(batch):
                qspan=item["schema_spans"]["question"]
                summary=model.question_summary_from_hidden(hidden[bi],qspan)
                summaries.append(summary)
                t=item["targets"]
                pt.append(t["projection"]);at.append(t["arity"]);ct.append(t["clauses"]);cont.append(t["continuation"])
                if t["direction"] in (1,2):
                    dir_s.append(summary);dir_t.append(t["direction"])

                schema=item["schema"]
                table=t["table"]
                if table in schema:
                    tables=list(schema)
                    fields=[
                        [[*tok.encode_ids(tab)]]+[[*tok.encode_ids(c)] for c in schema[tab]]
                        for tab in tables
                    ]
                    ts=model.score_schema_fields_from_question(hidden[bi],qspan,fields,"table")
                    target=tables.index(table)
                    losses_ground.append(F.cross_entropy(ts.unsqueeze(0),torch.tensor([target])))
                    table_meta.append((int(ts.detach().argmax()),target))

                    candidates=list(schema[table])
                    token_lists=[tok.encode_ids(c) for c in candidates]

                    # General semantic column supervision: projection, WHERE, ORDER.
                    targets_cols=[]
                    for c in t["projection_columns"]+[t["where_column"],t["order_column"]]:
                        if c is not None and c in candidates and c not in targets_cols:targets_cols.append(c)
                    for c in targets_cols:
                        sem=model.score_schema_fields_from_question(
                            hidden[bi],qspan,[[ids] for ids in token_lists],"column"
                        )
                        target=candidates.index(c)
                        losses_ground.append(0.35*F.cross_entropy(sem.unsqueeze(0),torch.tensor([target])))

                    if t["where_column"] in candidates:
                        ws=model.score_clause_columns_from_question(hidden[bi],qspan,token_lists,"where")
                        target=candidates.index(t["where_column"])
                        losses_ground.append(2.5*F.cross_entropy(ws.unsqueeze(0),torch.tensor([target])))
                        where_meta.append((int(ws.detach().argmax()),target))
                    if t["order_column"] in candidates:
                        os=model.score_clause_columns_from_question(hidden[bi],qspan,token_lists,"order")
                        target=candidates.index(t["order_column"])
                        losses_ground.append(2.5*F.cross_entropy(os.unsqueeze(0),torch.tensor([target])))
                        order_meta.append((int(os.detach().argmax()),target))

                    lower=[tok.encode_ids(c.lower()) for c in candidates]
                    if all(lower) and (any(c!=c.lower() for c in candidates) or t["where_column"] in candidates):
                        target_col=t["where_column"] or (t["projection_columns"][0] if t["projection_columns"] else None)
                        if target_col in candidates:
                            ns=model.score_normalized_columns_from_question(hidden[bi],qspan,lower)
                            target=candidates.index(target_col)
                            losses_ground.append(0.7*F.cross_entropy(ns.unsqueeze(0),torch.tensor([target])))
                            norm_meta.append((int(ns.detach().argmax()),target))

                # Actual generated-prefix contextual pointer decisions.
                for decision in item["contextual"]:
                    if decision["position"]<0 or decision["position"]>=hidden.shape[1]:continue
                    scores=model.score_schema_spans_from_hidden(
                        hidden[bi],decision["position"],decision["candidate_spans"],decision["kind"]
                    )
                    target=decision["target_index"]
                    losses_ground.append(0.7*F.cross_entropy(scores.unsqueeze(0),torch.tensor([target])))
                    ctx_meta.append((int(scores.detach().argmax()),target))

            summaries=torch.stack(summaries)
            pt=torch.tensor(pt,dtype=torch.long);at=torch.tensor(at,dtype=torch.long)
            ct=torch.stack(ct);cont=torch.tensor(cont,dtype=torch.long)

            pl=model.projection_head(summaries);al=model.projection_arity_head(summaries)
            cl=model.clause_head(summaries);col=model.continuation_head(summaries)
            p_loss=F.cross_entropy(pl,pt,weight=pweights)
            a_loss=F.cross_entropy(al,at,weight=aweights)
            c_loss=F.binary_cross_entropy_with_logits(cl,ct,pos_weight=cpos)
            cont_loss=F.cross_entropy(col,cont,weight=contweights)

            if dir_s:
                ds=torch.stack(dir_s);dt=torch.tensor(dir_t,dtype=torch.long)
                dl=model.direction_head(ds)
                d_loss=F.cross_entropy(dl,dt)
                totals["direction_correct"]+=int((dl.detach().argmax(-1)==dt).sum())
                totals["direction_total"]+=len(dt)
            else:d_loss=p_loss.new_zeros(())

            g_loss=torch.stack(losses_ground).mean() if losses_ground else p_loss.new_zeros(())
            loss=p_loss+a_loss+c_loss+1.5*cont_loss+0.8*d_loss+2.0*g_loss

            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()

            n=len(batch);totals["examples"]+=n;totals["loss"]+=float(loss.detach())*n
            totals["projection_correct"]+=int((pl.detach().argmax(-1)==pt).sum())
            totals["arity_correct"]+=int((al.detach().argmax(-1)==at).sum())
            totals["clause_correct"]+=int(((torch.sigmoid(cl.detach())>=0.5).float()==ct).sum())
            totals["clause_bits"]+=n*3
            totals["continuation_correct"]+=int((col.detach().argmax(-1)==cont).sum())
            for name,meta in (
                ("table",table_meta),("where",where_meta),("order",order_meta),
                ("normalized",norm_meta),("context",ctx_meta),
            ):
                totals[name+"_correct"]+=sum(int(a==b) for a,b in meta)
                totals[name+"_total"]+=len(meta)
            totals["ground_loss"]+=sum(float(x.detach()) for x in losses_ground)

        n=max(totals["examples"],1)
        metrics={
            "epoch":epoch,"source_examples":len(encoded),"effective_examples":len(effective),
            "mean_loss":totals["loss"]/n,
            "projection_accuracy":totals["projection_correct"]/n,
            "arity_accuracy":totals["arity_correct"]/n,
            "clause_bit_accuracy":totals["clause_correct"]/max(totals["clause_bits"],1),
            "direction_accuracy":totals["direction_correct"]/max(totals["direction_total"],1),
            "continuation_accuracy":totals["continuation_correct"]/n,
            "table_accuracy":totals["table_correct"]/max(totals["table_total"],1),
            "where_column_accuracy":totals["where_correct"]/max(totals["where_total"],1),
            "order_column_accuracy":totals["order_correct"]/max(totals["order_total"],1),
            "normalized_accuracy":totals["normalized_correct"]/max(totals["normalized_total"],1),
            "contextual_accuracy":totals["context_correct"]/max(totals["context_total"],1),
            "grounding_loss_sum":totals["ground_loss"],
        }
        print(json.dumps(metrics))
        all_metrics.append(metrics)
        torch.save(model.state_dict(),CHECKPOINT)
        torch.save(opt.state_dict(),OPTIMIZER)

    sql_loss,sql_tokens=sql_target_loss(model,encoded,tok)
    output={"seed":args.seed,"epochs":all_metrics,"final_sql_target_loss":sql_loss,"sql_target_tokens":sql_tokens}
    PROGRESS.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"final_sql_target_loss":sql_loss,"sql_target_tokens":sql_tokens}))

if __name__=="__main__":
    main()
