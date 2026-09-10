from __future__ import annotations
import math
import re
import torch
import torch.nn.functional as F

from text_to_sql.schema_role_encoder import SchemaRoleEncoder


def parse_targets(sql):
    upper=sql.upper().strip()
    if upper.startswith("SELECT DISTINCT"):projection=2
    elif upper.startswith("SELECT COUNT"):projection=3
    elif upper.startswith("SELECT AVG"):projection=4
    elif upper.startswith("SELECT *"):projection=0
    else:projection=1

    from_pos=upper.index(" FROM ")
    select_part=sql[len("SELECT "):from_pos].strip()
    if projection in (0,3):arity=0
    elif projection in (2,4):arity=1
    else:arity=2 if "," in select_part else 1

    where=" WHERE " in upper
    order=" ORDER BY " in upper
    limit=" LIMIT " in upper
    clauses=torch.tensor([float(where),float(order),float(limit)],dtype=torch.float32)

    if not where:operator=0
    elif re.search(r"\sIN\s*\(",upper):operator=4
    elif " > " in upper:operator=1
    elif " < " in upper:operator=2
    else:operator=3

    if " DESC" in upper:direction=2
    elif " ASC" in upper:direction=1
    else:direction=0

    continuation=3 if order and limit else 1 if order else 2 if limit else 0

    table_match=re.search(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)",sql,re.I)
    table=table_match.group(1) if table_match else None
    where_match=re.search(r"\bWHERE\s+([A-Za-z_][A-Za-z0-9_]*)",sql,re.I)
    where_col=where_match.group(1) if where_match else None
    order_match=re.search(r"\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)",sql,re.I)
    order_col=order_match.group(1) if order_match else None

    projection_columns=[]
    if projection==1:
        projection_columns=[part.strip() for part in select_part.split(",")]
    elif projection==2:
        projection_columns=[select_part[len("DISTINCT "):].strip()]
    elif projection==4:
        projection_columns=[select_part[4:-1].strip()]

    return {
        "projection":projection,"arity":arity,"clauses":clauses,"operator":operator,
        "direction":direction,"continuation":continuation,"table":table,
        "where_column":where_col,"order_column":order_col,
        "projection_columns":projection_columns,
    }


def _all_unique_columns(schema):
    out=[]
    for columns in schema.values():
        for c in columns:
            if c not in out:out.append(c)
    return out


def _span_groups_for_columns(schema,spans,candidates,table=None):
    groups=[]
    for column in candidates:
        group=[]
        if table is not None:
            span=spans["columns"].get((table,column))
            if span is not None:group.append(span)
        else:
            for t,cols in schema.items():
                if column in cols:
                    span=spans["columns"].get((t,column))
                    if span is not None:group.append(span)
        groups.append(group)
    return groups


def _table_span_groups(schema,spans):
    groups=[]
    for table,columns in schema.items():
        group=[]
        if table in spans["tables"]:group.append(spans["tables"][table])
        for c in columns:
            span=spans["columns"].get((table,c))
            if span is not None:group.append(span)
        groups.append(group)
    return groups


def encode_example(example,tokenizer):
    encoder=SchemaRoleEncoder(tokenizer)
    prompt_ids,prompt_roles,spans=encoder.encode_prompt_with_spans(example["schema"],example["question"])
    sql_ids=tokenizer.encode_ids(example["sql"])
    ids=prompt_ids+sql_ids
    roles=prompt_roles+[SchemaRoleEncoder.SQL]*len(sql_ids)
    targets=parse_targets(example["sql"])

    contextual=[]
    table=targets["table"]
    if table in example["schema"]:
        prefix=example["sql"][:re.search(r"\bFROM\s+",example["sql"],re.I).end()]
        contextual.append({
            "kind":"table",
            "position":len(prompt_ids)+len(tokenizer.encode_ids(prefix))-1,
            "candidate_spans":_table_span_groups(example["schema"],spans),
            "target_index":list(example["schema"]).index(table),
        })

        # Projection columns in textual order.
        from_start=example["sql"].upper().index(" FROM ")
        search_start=len("SELECT ")
        for column in targets["projection_columns"]:
            match=re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])",
                example["sql"][search_start:from_start],re.I
            )
            if match:
                absolute=search_start+match.start()
                candidates=_all_unique_columns(example["schema"])
                contextual.append({
                    "kind":"column",
                    "position":len(prompt_ids)+len(tokenizer.encode_ids(example["sql"][:absolute]))-1,
                    "candidate_spans":_span_groups_for_columns(example["schema"],spans,candidates),
                    "target_index":candidates.index(column),
                })
                search_start=absolute+len(column)

        # Clause columns use selected table candidates.
        for pattern,column in (
            (r"\bWHERE\s+([A-Za-z_][A-Za-z0-9_]*)",targets["where_column"]),
            (r"\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)",targets["order_column"]),
        ):
            if column is None:continue
            match=re.search(pattern,example["sql"],re.I)
            if not match:continue
            candidates=list(example["schema"][table])
            if column not in candidates:continue
            contextual.append({
                "kind":"column",
                "position":len(prompt_ids)+len(tokenizer.encode_ids(example["sql"][:match.start(1)]))-1,
                "candidate_spans":_span_groups_for_columns(example["schema"],spans,candidates,table),
                "target_index":candidates.index(column),
            })

    return {
        "input_ids":ids,"role_ids":roles,"prompt_length":len(prompt_ids),
        "schema_spans":spans,"targets":targets,"contextual":contextual,
        "schema":example["schema"],"question":example["question"],"sql":example["sql"],
        "category":example.get("category",""),
    }


def batchify(items,pad_id):
    n=max(len(x["input_ids"]) for x in items)
    ids=[];roles=[]
    for x in items:
        pad=n-len(x["input_ids"])
        ids.append(x["input_ids"]+[pad_id]*pad)
        roles.append(x["role_ids"]+[SchemaRoleEncoder.NORMAL]*pad)
    return torch.tensor(ids,dtype=torch.long),torch.tensor(roles,dtype=torch.long)


def load_legacy_initialization(model,checkpoint_path):
    state=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    current=model.state_dict()
    copied=[]
    for key,value in state.items():
        if key=="schema_role_embedding.weight":continue
        if key in current and current[key].shape==value.shape:
            current[key]=value
            copied.append(key)
    old=state.get("schema_role_embedding.weight")
    if old is not None:
        new=current["schema_role_embedding.weight"]
        count=min(old.shape[0],new.shape[0])
        new[:count]=old[:count]
        # QUESTION and SQL roles start distinct but close to legacy NORMAL.
        generator=torch.Generator().manual_seed(42)
        for i in range(count,new.shape[0]):
            noise=torch.randn(new[i].shape,generator=generator)*0.005
            new[i]=old[0]+noise
        current["schema_role_embedding.weight"]=new
    model.load_state_dict(current,strict=True)
    return copied


def trainable_aux_parameters(model):
    for p in model.parameters():p.requires_grad=False
    prefixes=(
        "question_pool_score","projection_head","projection_arity_head","clause_head",
        "direction_head","continuation_head",
        "column_query_projection","column_key_projection","table_query_projection","table_key_projection",
        "lexical_table_query_projection","lexical_table_key_projection",
        "lexical_column_query_projection","lexical_column_key_projection",
        "where_question_pool_score","order_question_pool_score",
        "where_column_query_projection","where_column_key_projection",
        "order_column_query_projection","order_column_key_projection",
        "normalized_column_query_projection","normalized_column_key_projection",
    )
    params=[]
    for name,p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad=True;params.append(p)
    return params


def sql_target_loss(model,items,tokenizer,sample_size=160):
    losses=0.0;tokens=0
    model.eval()
    with torch.no_grad():
        for item in items[:sample_size]:
            ids=item["input_ids"];roles=item["role_ids"];start=max(item["prompt_length"]-1,0)
            logits=model.forward(ids[:-1],roles[:-1])[0]
            targets=torch.tensor(ids[1:],dtype=torch.long)
            if start>=len(targets):continue
            losses+=float(F.cross_entropy(logits[start:],targets[start:],reduction="sum"))
            tokens+=len(targets)-start
    return losses/max(tokens,1),tokens
