import sqlite3

from evaluation.run_v2_regression_suite import load
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline


def test_numeric_comparison_uses_type_compatible_vehicle_column():
    schema={
        "vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
        "owners":["id","name","age","membership_level"],
    }
    column_types={
        ("vehicles","id"):"INTEGER",
        ("vehicles","brand"):"TEXT",
        ("vehicles","model"):"TEXT",
        ("vehicles","year"):"INTEGER",
        ("vehicles","price"):"REAL",
        ("vehicles","mileage"):"INTEGER",
        ("vehicles","fuel_type"):"TEXT",
        ("vehicles","rating"):"REAL",
        ("owners","id"):"INTEGER",
        ("owners","name"):"TEXT",
        ("owners","age"):"INTEGER",
        ("owners","membership_level"):"TEXT",
    }
    tokenizer,model,semantic_encoder=load()
    connection=sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE vehicles (id INTEGER, brand TEXT, model TEXT, year INTEGER, price REAL, mileage INTEGER, fuel_type TEXT, rating REAL)")
    connection.execute("CREATE TABLE owners (id INTEGER, name TEXT, age INTEGER, membership_level TEXT)")
    pipeline=TextToSQLPipeline(
        model,tokenizer,schema,connection,beam_width=12,
        semantic_schema_encoder=semantic_encoder,schema_semantic_weight=4.0,
        column_types=column_types,
    )
    try:
        expected={
            "vehicles released after 2014":"SELECT * FROM vehicles WHERE year > 2014;",
            "vehicles released after 2018":"SELECT * FROM vehicles WHERE year > 2018;",
            "vehicles released after 2019":"SELECT * FROM vehicles WHERE year > 2019;",
            "vehicles released after 2025":"SELECT * FROM vehicles WHERE year > 2025;",
        }
        actual={question:pipeline.generate(question) for question in expected}
        assert actual==expected
    finally:
        connection.close()


def test_runtime_passes_database_column_types_to_grounding(tmp_path,monkeypatch):
    from text_to_sql_runtime import TextToSQLRuntime

    db_path=tmp_path/"vehicles.db"
    connection=sqlite3.connect(db_path)
    connection.execute("CREATE TABLE vehicles (id INTEGER, brand TEXT, model TEXT, year INTEGER, price REAL, mileage INTEGER, fuel_type TEXT, rating REAL)")
    connection.execute("CREATE TABLE owners (id INTEGER, name TEXT, age INTEGER, membership_level TEXT)")
    connection.commit()
    connection.close()

    monkeypatch.setenv("DB_ENGINE","sqlite")
    monkeypatch.setenv("DB_PATH",str(db_path))
    runtime=TextToSQLRuntime()
    try:
        for year in (2018,2019,2025):
            assert runtime.generate(
                f"vehicles released after {year}"
            )==f"SELECT * FROM vehicles WHERE year > {year};"
        assert runtime.generate(
            "show vehicles with mileage greater than 2025"
        )=="SELECT * FROM vehicles WHERE mileage > 2025;"
    finally:
        runtime.close()
