from __future__ import annotations
import argparse,sqlite3,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline

UNSEEN=[
({"customers":["id","name","age"]},
 "show customers where age is greater than 30",
 "SELECT * FROM customers WHERE age > 30;"),
({"customers":["id","name","city","age","membership_level"]},
 "show customers older than 30",
 "SELECT * FROM customers WHERE age > 30;"),
({"students":["ID","Name","Age"]},
 "show all students older than 15",
 "SELECT * FROM students WHERE Age > 15;"),
({"students":["ID","Name","Age"]},
 "show students older than 15",
 "SELECT * FROM students WHERE Age > 15;"),
]

HELDOUT=[
({"employees":["id","name","department","salary","age","years_experience"],
  "products":["id","name","price","stock","rating"]},
 "show records with experience more than 5 years",
 "SELECT * FROM employees WHERE years_experience > 5;"),
({"employees":["id","name","department","salary","age","years_experience"],
  "products":["id","name","price","stock","rating"]},
 "people having over 5 years experience",
 "SELECT * FROM employees WHERE years_experience > 5;"),
({"consultants":["consultant_id","consultant_name","work_experience","hourly_rate"],
  "departments":["department_id","department_name","budget"]},
 "show records with work experience more than 10 years",
 "SELECT * FROM consultants WHERE work_experience > 10;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
  "owners":["id","name","age","membership_level"]},
 "show records priced below 20000",
 "SELECT * FROM vehicles WHERE price < 20000;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
  "owners":["id","name","age","membership_level"]},
 "show vehicles with mileage greater than 30000",
 "SELECT * FROM vehicles WHERE mileage > 30000;"),
({"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
  "owners":["id","name","age","membership_level"]},
 "show records with rating above 4.5",
 "SELECT * FROM vehicles WHERE rating > 4.5;"),
]

def make_model():
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(
        torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),
        strict=True,
    )
    model.eval()
    return tok,model

def run(cases):
    tok,model=make_model()
    passed=0
    for i,(schema,q,expected) in enumerate(cases):
        conn=sqlite3.connect(":memory:")
        for table,columns in schema.items():
            defs=", ".join(f'"{c}" REAL' for c in columns)
            conn.execute(f'CREATE TABLE "{table}" ({defs})')
        pipe=TextToSQLPipeline(model,tok,schema,conn,beam_width=12)
        try:actual=pipe.generate(q)
        except Exception as e:actual=f"<ERR {type(e).__name__}: {e}>"
        ok=actual==expected;passed+=int(ok)
        print(i,"PASS" if ok else "FAIL",actual,"EXPECTED="+expected)
        conn.close()
    print(f"SUMMARY {passed}/{len(cases)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite",choices=["unseen","heldout"],required=True)
    args=ap.parse_args()
    run(UNSEEN if args.suite=="unseen" else HELDOUT)

if __name__=="__main__":main()
