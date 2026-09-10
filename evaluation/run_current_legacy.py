from __future__ import annotations
import argparse,ast,sqlite3,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline

def cases():
    tree=ast.parse((ROOT/"tests/test_option_a_regressions.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=="test_option_a_real_model_regressions":
            for dec in node.decorator_list:
                if isinstance(dec,ast.Call) and getattr(dec.func,"attr",None)=="parametrize":
                    return list(ast.literal_eval(dec.args[1]))
    return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",type=int,required=True)
    ap.add_argument("--end",type=int,required=True)
    args=ap.parse_args()
    schema={
        "employees":["id","name","department","salary","age","years_experience"],
        "products":["id","name","category","price","stock","rating"],
        "orders":["id","product_id","employee_id","quantity","total_amount","status"],
    }
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),strict=True)
    model.eval()
    conn=sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE employees (id INTEGER,name TEXT,department TEXT,salary REAL,age INTEGER,years_experience INTEGER)")
    conn.execute("CREATE TABLE products (id INTEGER,name TEXT,category TEXT,price REAL,stock INTEGER,rating REAL)")
    conn.execute("CREATE TABLE orders (id INTEGER,product_id INTEGER,employee_id INTEGER,quantity INTEGER,total_amount REAL,status TEXT)")
    pipe=TextToSQLPipeline(model,tok,schema,conn,beam_width=12)
    pairs=cases()
    passed=0
    for i in range(args.start,min(args.end,len(pairs))):
        q,expected=pairs[i]
        try:actual=pipe.generate(q)
        except Exception as e:actual=f"<ERR {type(e).__name__}: {e}>"
        ok=actual==expected;passed+=int(ok)
        print(i,"PASS" if ok else "FAIL",actual,"EXPECTED="+expected)
    print(f"SUMMARY {passed}/{min(args.end,len(pairs))-args.start}")
    conn.close()

if __name__=="__main__":main()
