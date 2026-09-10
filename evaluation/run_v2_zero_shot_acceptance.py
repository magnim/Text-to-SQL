from __future__ import annotations
import sqlite3,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2

CASES=[
({"cinema_catalog":["id","title","release_year","rating"],"store_items":["id","name","price","stock"],"accounts":["id","name","age","income"]},
 "released after 2014","SELECT * FROM cinema_catalog WHERE release_year > 2014;"),
({"meal_plans":["id","dish_name","preparation_time","calories","rating"],"store_items":["id","name","price","stock"],"accounts":["id","name","age","income"]},
 "recipes that take more than 15 mins to prepare","SELECT * FROM meal_plans WHERE preparation_time > 15;"),
({"air_routes":["id","route_code","duration_minutes","fare","stops"],"hotel_rooms":["id","room_name","nightly_rate","rating"]},
 "flights lasting more than 180 minutes","SELECT * FROM air_routes WHERE duration_minutes > 180;"),
({"academic_institutions":["id","institution_name","founded_year","enrollment","rating"],"hotel_rooms":["id","room_name","nightly_rate","rating"]},
 "universities founded before 1900","SELECT * FROM academic_institutions WHERE founded_year < 1900;"),
({"lodging_options":["id","property_name","nightly_rate","rating","capacity"],"retail_catalog":["id","item_name","unit_price","stock"]},
 "hotels costing less than 100 per night","SELECT * FROM lodging_options WHERE nightly_rate < 100;"),
]

def load():
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    m=TinyGPT(len(tok.vocab),64,512,4,128,4,5);m.load_state_dict(torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),strict=True);m.eval()
    s=SchemaSemanticEncoderV2(len(tok.vocab),64,64,4,160,2,tok.vocab["<PAD>"]);s.load_state_dict(torch.load(ROOT/"semantic_schema_encoder_v2.pt",map_location="cpu",weights_only=True),strict=True);s.eval()
    return tok,m,s

def main():
    tok,m,s=load();passed=0
    for i,(schema,q,e) in enumerate(CASES):
        c=sqlite3.connect(":memory:")
        for t,cols in schema.items():c.execute(f'CREATE TABLE "{t}" ({", ".join(f"""\"{x}\" REAL""" for x in cols)})')
        p=TextToSQLPipeline(m,tok,schema,c,beam_width=12,semantic_schema_encoder=s,schema_semantic_weight=4.0)
        try:a=p.generate(q)
        except Exception as ex:a=f"<ERROR:{type(ex).__name__}>"
        ok=a==e;passed+=int(ok);print(i,"PASS" if ok else "FAIL",a,"EXPECTED="+e);c.close()
    print(f"SUMMARY {passed}/{len(CASES)}")
if __name__=="__main__":main()
