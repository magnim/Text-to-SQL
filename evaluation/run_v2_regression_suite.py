from __future__ import annotations
import argparse,ast,sqlite3,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
LEGACY_SCHEMA={"employees":["id","name","department","salary","age","years_experience"],"products":["id","name","category","price","stock","rating"],"orders":["id","product_id","employee_id","quantity","total_amount","status"]}
UNSEEN=[
({"customers":["id","name","age"]},"show customers where age is greater than 30","SELECT * FROM customers WHERE age > 30;"),
({"customers":["id","name","city","age","membership_level"]},"show customers older than 30","SELECT * FROM customers WHERE age > 30;"),
({"students":["ID","Name","Age"]},"show all students older than 15","SELECT * FROM students WHERE Age > 15;"),
({"students":["ID","Name","Age"]},"show students older than 15","SELECT * FROM students WHERE Age > 15;")]
HELDOUT=[
({"employees":["id","name","department","salary","age","years_experience"],"products":["id","name","price","stock","rating"]},"show records with experience more than 5 years","SELECT * FROM employees WHERE years_experience > 5;"),
({"employees":["id","name","department","salary","age","years_experience"],"products":["id","name","price","stock","rating"]},"people having over 5 years experience","SELECT * FROM employees WHERE years_experience > 5;"),
({"consultants":["consultant_id","consultant_name","work_experience","hourly_rate"],"departments":["department_id","department_name","budget"]},"show records with work experience more than 10 years","SELECT * FROM consultants WHERE work_experience > 10;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],"owners":["id","name","age","membership_level"]},"show records priced below 20000","SELECT * FROM vehicles WHERE price < 20000;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],"owners":["id","name","age","membership_level"]},"show vehicles with mileage greater than 30000","SELECT * FROM vehicles WHERE mileage > 30000;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],"owners":["id","name","age","membership_level"]},"show records with rating above 4.5","SELECT * FROM vehicles WHERE rating > 4.5;")]
def legacy_cases():
    tree=ast.parse((ROOT/"tests/test_option_a_regressions.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=="test_option_a_real_model_regressions":
            for d in n.decorator_list:
                if isinstance(d,ast.Call) and getattr(d.func,"attr",None)=="parametrize":return list(ast.literal_eval(d.args[1]))
    return []
def load():
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    m=TinyGPT(len(tok.vocab),64,512,4,128,4,5);m.load_state_dict(torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),strict=True);m.eval()
    s=SchemaSemanticEncoderV2(len(tok.vocab),64,64,4,160,2,tok.vocab["<PAD>"]);s.load_state_dict(torch.load(ROOT/"semantic_schema_encoder_v2.pt",map_location="cpu",weights_only=True),strict=True);s.eval()
    return tok,m,s
def run(cases,start,end):
    tok,m,s=load();passed=0;stop=min(end,len(cases))
    for i in range(start,stop):
        item=cases[i]
        if len(item)==2:q,e=item;schema=LEGACY_SCHEMA
        else:schema,q,e=item
        c=sqlite3.connect(":memory:")
        for t,cols in schema.items():c.execute(f'CREATE TABLE "{t}" ({", ".join(f"""\"{x}\" REAL""" for x in cols)})')
        p=TextToSQLPipeline(m,tok,schema,c,beam_width=12,semantic_schema_encoder=s,schema_semantic_weight=4.0)
        try:a=p.generate(q)
        except Exception as ex:a=f"<ERROR:{type(ex).__name__}>"
        ok=a==e;passed+=int(ok);print(i,"PASS" if ok else "FAIL",a,"EXPECTED="+e);c.close()
    print(f"SUMMARY {passed}/{stop-start}")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--suite",choices=["legacy","unseen","heldout"],required=True);ap.add_argument("--start",type=int,default=0);ap.add_argument("--end",type=int,default=999);a=ap.parse_args()
    run({"legacy":legacy_cases(),"unseen":UNSEEN,"heldout":HELDOUT}[a.suite],a.start,a.end)
if __name__=="__main__":main()
