import re
import sqlite3
from pathlib import Path

import pytest
import torch

from pytorch_impl.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from text_to_sql.sql_grammar import SQLGrammar
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2
from training.schema_conditioned_dataset import build_schema_conditioned_examples


@pytest.fixture(scope="module")
def tokenizer_model():
    root=Path(__file__).resolve().parents[1]
    tokenizer=BPETrainer.load(str(root/"tokenizer_balanced.json"))
    model=TinyGPT(
        len(tokenizer.vocab),64,512,4,128,4,5
    )
    model.load_state_dict(
        torch.load(
            root/"tiny_gpt_schema_aware_final.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    model.eval()
    semantic=SchemaSemanticEncoderV2(
        len(tokenizer.vocab),64,64,4,160,2,tokenizer.vocab["<PAD>"]
    )
    semantic.load_state_dict(
        torch.load(
            root/"semantic_schema_encoder_v2.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    semantic.eval()
    return tokenizer,model,semantic


def make_pipeline(tokenizer,model,semantic,schema):
    connection=sqlite3.connect(":memory:")
    for table,columns in schema.items():
        definitions=", ".join(
            f'"{column}" REAL' for column in columns
        )
        connection.execute(
            f'CREATE TABLE "{table}" ({definitions})'
        )
    return connection,TextToSQLPipeline(
        model,tokenizer,schema,connection,beam_width=12,
        semantic_schema_encoder=semantic,schema_semantic_weight=4.0
    )


def test_schema_roles_are_five(tokenizer_model):
    tokenizer,_,_=tokenizer_model
    encoder=SchemaRoleEncoder(tokenizer)
    assert (
        encoder.NORMAL,
        encoder.TABLE,
        encoder.COLUMN,
        encoder.QUESTION,
        encoder.SQL,
        encoder.NUMBER_OF_ROLES,
    )==(0,1,2,3,4,5)


def test_complete_schema_precedes_question(tokenizer_model):
    tokenizer,_,_=tokenizer_model
    schema={
        "alpha_records":["id","metric_a"],
        "beta_records":["id","metric_b"],
    }
    encoder=SchemaRoleEncoder(tokenizer)
    ids,roles,spans=encoder.encode_prompt_with_spans(
        schema,"show records where metric b is above 5"
    )
    q_start,q_end=spans["question"]
    assert spans["tables"]["alpha_records"][1]<=q_start
    assert spans["tables"]["beta_records"][1]<=q_start
    assert spans["columns"][("alpha_records","metric_a")][1]<=q_start
    assert spans["columns"][("beta_records","metric_b")][1]<=q_start
    assert all(
        role==encoder.QUESTION
        for role in roles[q_start:q_end]
    )
    assert len(ids)==len(roles)


@pytest.mark.parametrize(
    "identifier",
    [
        "years_experience",
        "work_experience",
        "membership_level",
        "consultant_name",
        "fuel_type",
    ],
)
def test_bpe_schema_identifier_round_trip(
    tokenizer_model,identifier
):
    tokenizer,_,_=tokenizer_model
    ids=tokenizer.encode_ids(identifier)
    assert tokenizer.vocab["<UNK>"] not in ids
    assert tokenizer.decode_ids(ids)==identifier


def test_compositional_grammar():
    grammar=SQLGrammar()
    sequence=[
        ("SELECT",None),("*",None),("FROM",None),("TABLE",None),
        ("WHERE",None),("COLUMN",None),("OPERATOR",">"),
        ("VALUE",None),("ORDER BY",None),("COLUMN",None),
        ("DIRECTION",None),("LIMIT",None),("VALUE",None),(";",None),
    ]
    for token_type,candidate in sequence:
        grammar.consume(token_type,candidate=candidate)
    assert grammar.state==SQLGrammar.COMPLETE


def test_production_inference_has_no_phrase_to_schema_rules():
    root=Path(__file__).resolve().parents[1]
    production=[
        root/"text_to_sql/text_to_sql_pipeline.py",
        root/"text_to_sql/sql_beam.py",
        root/"text_to_sql/constrained_decoder.py",
        root/"text_to_sql/schema_role_encoder.py",
    ]
    source="\n".join(p.read_text(encoding="utf-8") for p in production)
    forbidden=[
        r'if\s+["\']experience["\']\s+in',
        r'if\s+["\']mileage["\']\s+in',
        r'if\s+["\']rating["\']\s+in',
        r'if\s+["\']price["\']\s+in',
        r'SchemaLinker\s*\(',
        r'SemanticColumnLinker\s*\(',
        r'SemanticColumnRanker\s*\(',
    ]
    assert not any(
        re.search(pattern,source,re.I)
        for pattern in forbidden
    )


def test_heldout_schema_names_absent_from_training():
    examples=build_schema_conditioned_examples(42)
    source="\n".join(
        example["question"]+" "+example["sql"]+" "
        +" ".join(example["schema"])
        for example in examples
    ).lower()
    assert "consultants" not in source
    assert "vehicles" not in source


HELDOUT=[
(
{"employees":["id","name","department","salary","age","years_experience"],
 "products":["id","name","price","stock","rating"]},
"show records with experience more than 5 years",
"SELECT * FROM employees WHERE years_experience > 5;",
),
(
{"employees":["id","name","department","salary","age","years_experience"],
 "products":["id","name","price","stock","rating"]},
"people having over 5 years experience",
"SELECT * FROM employees WHERE years_experience > 5;",
),
(
{"consultants":["consultant_id","consultant_name","work_experience","hourly_rate"],
 "departments":["department_id","department_name","budget"]},
"show records with work experience more than 10 years",
"SELECT * FROM consultants WHERE work_experience > 10;",
),
(
{"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
 "owners":["id","name","age","membership_level"]},
"show records priced below 20000",
"SELECT * FROM vehicles WHERE price < 20000;",
),
(
{"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
 "owners":["id","name","age","membership_level"]},
"show vehicles with mileage greater than 30000",
"SELECT * FROM vehicles WHERE mileage > 30000;",
),
(
{"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
 "owners":["id","name","age","membership_level"]},
"show records with rating above 4.5",
"SELECT * FROM vehicles WHERE rating > 4.5;",
),
]


@pytest.mark.parametrize(("schema","question","expected"),HELDOUT)
def test_heldout_semantic_generalization(
    tokenizer_model,schema,question,expected
):
    tokenizer,model,semantic=tokenizer_model
    connection,pipeline=make_pipeline(
        tokenizer,model,semantic,schema
    )
    try:
        assert pipeline.generate(question)==expected
    finally:
        connection.close()


COMPOSITIONS=[
(
"show employees where age is greater than 30 ordered by salary descending limited to 5",
"SELECT * FROM employees WHERE age > 30 ORDER BY salary DESC LIMIT 5;",
),
(
"show products where price is greater than 10 ordered by rating descending limited to 5",
"SELECT * FROM products WHERE price > 10 ORDER BY rating DESC LIMIT 5;",
),
(
"show orders where quantity is greater than 2 ordered by total amount descending limited to 5",
"SELECT * FROM orders WHERE quantity > 2 ORDER BY total_amount DESC LIMIT 5;",
),
]


@pytest.mark.parametrize(("question","expected"),COMPOSITIONS)
def test_where_order_limit_composition(
    tokenizer_model,question,expected
):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "employees":["id","name","department","salary","age","years_experience"],
        "products":["id","name","category","price","stock","rating"],
        "orders":["id","product_id","employee_id","quantity","total_amount","status"],
    }
    connection,pipeline=make_pipeline(
        tokenizer,model,semantic,schema
    )
    try:
        sql=pipeline.generate(question)
        assert sql==expected
        connection.execute(sql)
    finally:
        connection.close()


ZERO_SHOT_ORDER=[
(
{"movies":["id","title","release_year","rating"],
 "accounts":["id","name","age","income"]},
"show movies ordered by release year descending",
"SELECT * FROM movies ORDER BY release_year DESC;",
),
(
{"vehicles":["id","brand","model","mileage","price","rating"],
 "owners":["id","name","age"]},
"show vehicles ordered by mileage descending",
"SELECT * FROM vehicles ORDER BY mileage DESC;",
),
(
{"meal_plans":["id","dish_name","preparation_time","calories","rating"],
 "accounts":["id","name","age"]},
"show recipes ordered by preparation time descending",
"SELECT * FROM meal_plans ORDER BY preparation_time DESC;",
),
]


@pytest.mark.parametrize(("schema","question","expected"),ZERO_SHOT_ORDER)
def test_zero_shot_order_column_grounding(
    tokenizer_model,schema,question,expected
):
    tokenizer,model,semantic=tokenizer_model
    connection,pipeline=make_pipeline(
        tokenizer,model,semantic,schema
    )
    try:
        assert pipeline.generate(question)==expected
    finally:
        connection.close()


MOVIE_TEMPORAL_REGRESSIONS=[
(
"movies released after 2014",
"SELECT * FROM movies WHERE release_year > 2014;",
),
(
"movies release year after 2014",
"SELECT * FROM movies WHERE release_year > 2014;",
),
]


@pytest.mark.parametrize(
    ("question","expected"),
    MOVIE_TEMPORAL_REGRESSIONS,
)
def test_movie_temporal_value_role_regression(
    tokenizer_model,question,expected
):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "movies":["id","title","release_year","rating"],
    }
    connection,pipeline=make_pipeline(
        tokenizer,model,semantic,schema
    )
    try:
        sql=pipeline.generate(question)
        assert sql==expected
        assert "LIMIT 2014" not in sql
        connection.execute(sql)
    finally:
        connection.close()


def test_single_numeric_literal_prefers_learned_where_role_over_limit(
    tokenizer_model,
):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "movies":["id","title","release_year","rating"],
        "users":["id","name","age","income"],
        "orders":["id","user_id","amount","status"],
        "products":["id","name","price","rating"],
    }
    connection,pipeline=make_pipeline(tokenizer,model,semantic,schema)
    try:
        assert pipeline.generate(
            "movies release year after 2014"
        )=="SELECT * FROM movies WHERE release_year > 2014;"
        assert pipeline.generate(
            "return only 5 movies"
        )=="SELECT * FROM movies LIMIT 5;"
    finally:
        connection.close()


NOISY_MOVIE_WHERE_REGRESSIONS=[
(
"movies released after 2019",
"SELECT * FROM movies WHERE release_year > 2019;",
),
(
"movies release year after 2019",
"SELECT * FROM movies WHERE release_year > 2019;",
),
(
"movies released before 2019",
"SELECT * FROM movies WHERE release_year < 2019;",
),
]


@pytest.mark.parametrize(("question","expected"),NOISY_MOVIE_WHERE_REGRESSIONS)
def test_noisy_movie_simple_where_grounding(
    tokenizer_model,question,expected
):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "movies":["id","title","director","release_year","rating"],
        "accounts":["id","name","age","income"],
        "products":["id","name","category","price","stock","rating"],
        "orders":["id","product_id","employee_id","quantity","total_amount","status"],
        "employees":["id","name","department","salary","age","years_experience"],
    }
    connection,pipeline=make_pipeline(tokenizer,model,semantic,schema)
    try:
        sql=pipeline.generate(question)
        assert sql==expected
        connection.execute(sql)
    finally:
        connection.close()


def test_movie_operator_is_invariant_to_unrelated_schema_tables(tokenizer_model):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "movies":["id","title","release_year","rating"],
        "accounts":["id","name","age","income"],
        "products":["id","name","category","price","stock","rating"],
        "orders":["id","product_id","employee_id","quantity","total_amount","status"],
        "employees":["id","name","department","salary","age","years_experience"],
        "vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
    }
    connection,pipeline=make_pipeline(tokenizer,model,semantic,schema)
    try:
        assert pipeline.generate(
            "movies release year after 2014"
        )=="SELECT * FROM movies WHERE release_year > 2014;"
    finally:
        connection.close()


def test_large_noisy_schema_reserves_generation_headroom(tokenizer_model):
    tokenizer,model,semantic=tokenizer_model
    schema={
        "podcasts":["id","title","release_year","host","duration","rating"],
    }
    noise_columns=[
        "record_id","record_name","created_year","owner_name","category_code","metric_value",
    ]
    for index in range(8):
        schema[f"archive_records_{index}"]=list(noise_columns)
    schema["archive_records_7"].extend(
        ["auxiliary_metric_0","auxiliary_metric_1"]
    )

    encoder=SchemaRoleEncoder(tokenizer)
    full_ids,_,_=encoder.encode_prompt_with_spans(
        schema,"podcasts released after 2019"
    )
    assert len(full_ids)<model.maximum_sequence_length
    assert len(full_ids)>model.maximum_sequence_length-32

    connection,pipeline=make_pipeline(tokenizer,model,semantic,schema)
    try:
        assert pipeline.generate(
            "podcasts released after 2019"
        )=="SELECT * FROM podcasts WHERE release_year > 2019;"
    finally:
        connection.close()
