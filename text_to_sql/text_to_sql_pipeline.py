from text_to_sql.schema_encoder import SchemaEncoder
from text_to_sql.prompt_builder import PromptBuilder
from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from text_to_sql.sql_grammar import SQLGrammar
from text_to_sql.execution_validator import ExecutionValidator
from text_to_sql.constrained_decoder import ConstrainedDecoder
from text_to_sql.sql_beam import SQLBeam, SQLBeamSearch
import re


class TextToSQLPipeline:
    def __init__(self, model, tokenizer, schema: dict[str, list[str]], connection, beam_width: int = 3) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.schema = schema
        self.connection = connection
        self.beam_width = beam_width
        self.schema_encoder = SchemaEncoder()
        self.prompt_builder = PromptBuilder()

    def generate(self, question: str) -> str:
        schema_text = self.schema_encoder.encode_schema(self.schema)
        prompt = self.prompt_builder.build_inference_prompt(schema_text=schema_text, question=question)

        input_ids = self.tokenizer.encode_ids(prompt)

        grammar = SQLGrammar()
        validator = ExecutionValidator(schema=self.schema, connection=self.connection)
        value_candidates = re.findall(r"\b\d+(?:\.\d+)?\b", question)
        decoder = ConstrainedDecoder(tokenizer=self.tokenizer, schema=self.schema, grammar=grammar,
                                     execution_validator=validator, value_candidates=value_candidates)

        initial_beam = SQLBeam(token_ids=input_ids, score=0.0, decoder=decoder, prompt_length=len(input_ids))
        beam_search = SQLBeamSearch(model=self.model, decoder=decoder, beam_width=self.beam_width)

        beams = beam_search.search(initial_beam=initial_beam)
        best_beam = beam_search.select_best_executable(beams)

        generated_ids = best_beam.token_ids[len(input_ids):]
        return self.tokenizer.decode_ids(generated_ids)