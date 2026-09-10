from __future__ import annotations
import argparse,sqlite3,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline

CASES=[
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",type=int,required=True)
    ap.add_argument("--end",type=int,required=True)
    args=ap.parse_args()

    torch.manual_seed(42)
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.eval()

    passed=0
    for i in range(args.start,min(args.end,len(CASES))):
        schema,q,expected=CASES[i]
        conn=sqlite3.connect(":memory:")
        for table,columns in schema.items():
            defs=", ".join(f'"{c}" REAL' for c in columns)
            conn.execute(f'CREATE TABLE "{table}" ({defs})')
        pipe=TextToSQLPipeline(model,tok,schema,conn,beam_width=12)
        try:
            actual=pipe.generate(q)
        except Exception as exc:
            actual=f"<ERROR:{type(exc).__name__}>"
        ok=actual==expected
        passed+=int(ok)
        print(i,"PASS" if ok else "FAIL",actual,"EXPECTED="+expected)
        conn.close()
    print(f"SUMMARY {passed}/{min(args.end,len(CASES))-args.start}")

if __name__=="__main__":
    main()
