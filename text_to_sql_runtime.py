from pathlib import Path
import torch

from database.connection import get_connection
from database.schema_scanner import scan_schema
from pytorch_impl.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2

PROJECT_ROOT=Path(__file__).resolve().parent
MODEL_PATH=PROJECT_ROOT/"tiny_gpt_schema_aware_final.pt"
TOKENIZER_PATH=PROJECT_ROOT/"tokenizer_balanced.json"
SEMANTIC_MODEL_PATH=PROJECT_ROOT/"semantic_schema_encoder_v2.pt"

class TextToSQLRuntime:
    def __init__(self):
        self.connection=get_connection()
        self.schema=scan_schema(self.connection)
        self.tokenizer=BPETrainer.load(str(TOKENIZER_PATH))
        self.model=TinyGPT(
            len(self.tokenizer.vocab),64,512,4,128,4,5
        )
        self.model.load_state_dict(
            torch.load(MODEL_PATH,map_location="cpu",weights_only=True),strict=True
        )
        self.model.eval()
        self.semantic_schema_encoder=SchemaSemanticEncoderV2(
            len(self.tokenizer.vocab),64,64,4,160,2,self.tokenizer.vocab["<PAD>"]
        )
        self.semantic_schema_encoder.load_state_dict(
            torch.load(SEMANTIC_MODEL_PATH,map_location="cpu",weights_only=True),strict=True
        )
        self.semantic_schema_encoder.eval()
        self.pipeline=self._build_pipeline()

    def _build_pipeline(self):
        return TextToSQLPipeline(
            self.model,self.tokenizer,self.schema,self.connection,beam_width=12,
            semantic_schema_encoder=self.semantic_schema_encoder,schema_semantic_weight=4.0
        )

    def generate(self,question):
        return self.pipeline.generate(question)

    def execute(self,sql):
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
        cursor=self.connection.cursor()
        try:
            cursor.execute(sql)
            rows=cursor.fetchall()
            return [dict(r) if not isinstance(r,dict) else r for r in rows]
        finally:
            cursor.close()

    def refresh_schema(self):
        old_connection=self.connection
        old_schema=self.schema
        old_pipeline=self.pipeline
        new_connection=get_connection()
        try:
            new_schema=scan_schema(new_connection)
            self.connection=new_connection
            self.schema=new_schema
            self.pipeline=self._build_pipeline()
        except Exception:
            self.connection=old_connection
            self.schema=old_schema
            self.pipeline=old_pipeline
            new_connection.close()
            raise
        try:old_connection.close()
        except Exception:pass
        return self.schema

    def close(self):
        if self.connection is not None:
            try:self.connection.close()
            finally:self.connection=None

if __name__=="__main__":
    runtime=TextToSQLRuntime()
    try:
        q="show unique product categories"
        print("Question:",q)
        print("Generated SQL:",runtime.generate(q))
    finally:
        runtime.close()
