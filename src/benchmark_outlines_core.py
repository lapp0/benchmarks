import json

from outlines.models.transformers import TransformerTokenizer
from outlines_core.fsm.guide import RegexGuide
from outlines_core.fsm.json_schema import build_regex_from_schema
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class OutlinesCoreRegex:
    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)

    def time_outlines_core(self, _, regex_name):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        regex_string = regex_cases[regex_name]["regex"]
        regex_samples = regex_cases[regex_name]["samples"]

        guide = RegexGuide(regex_string, self.tokenizer)

        for regex_sample in regex_samples:
            regex_sample_tokens = self.tokenizer.encode(regex_sample)[0][0]
            state = guide.initial_state
            for token in regex_sample_tokens:
                _ = guide.get_next_instruction(state)
                state = guide.get_next_state(state, token)


class OutlinesCoreJsonSchema:
    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)

    def time_outlines_core(self, _, json_schema_name):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        json_string = json_cases[json_schema_name]["schema"]
        json_samples = json_cases[json_schema_name]["samples"]

        regex_string = build_regex_from_schema(json.dumps(json_string))
        guide = RegexGuide(regex_string, self.tokenizer)

        for json_sample in json_samples:
            json_sample_tokens = self.tokenizer.encode(json_sample)[0][0]
            state = guide.initial_state
            for token in json_sample_tokens:
                _ = guide.get_next_instruction(state)
                state = guide.get_next_state(state, token)
