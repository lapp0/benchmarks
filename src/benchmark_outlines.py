"""Benchmark the Outlines library."""
import json

import outlines.caching as caching
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.models.transformers import TransformerTokenizer
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class OutlinesRegex:
    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 1200

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)
        RegexGuide("a", self.tokenizer)  # JIT-compile and convert the vocabulary

    def time_outlines(self, _, regex_name):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        caching.clear_cache()

        regex_string = regex_cases[regex_name]["regex"]
        regex_samples = regex_cases[regex_name]["samples"]

        guide = RegexGuide(regex_string, self.tokenizer)

        for regex_sample in regex_samples:
            regex_sample_tokens = self.tokenizer.encode(regex_sample)[0][0]
            state = guide.initial_state
            for token in regex_sample_tokens:
                _ = guide.get_next_instruction(state)
                state = guide.get_next_state(state, token)

    def teardown(self, *args):
        caching.clear_cache()


class OutlinesJsonSchema:
    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]

    timeout = 1200

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)
        RegexGuide("a", self.tokenizer)  # JIT-compile and convert the vocabulary

    def time_outlines(self, _, json_schema_name):
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

    def teardown(self, *args):
        caching.clear_cache()
