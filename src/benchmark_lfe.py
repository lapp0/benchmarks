"""Benchmark the lm-format-enforcer library."""
from lmformatenforcer import JsonSchemaParser, RegexParser, TokenEnforcer
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
)
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class LMFormatEnforcerRegex:
    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We convert the tokenizer during set up as this only
        needs to be done once for a given model.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

    def time_lfe(self, _, regex_name):
        regex_string = regex_cases[regex_name]["regex"]
        regex_samples = regex_cases[regex_name]["samples"]

        parser = RegexParser(regex_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for regex_sample in regex_samples:
            regex_sample_tokens = self.tokenizer.encode(regex_sample)
            for i in range(len(regex_sample_tokens)):
                _ = token_enforcer.get_allowed_tokens(regex_sample_tokens[: i + 1])


class LMFormatEnforcerJsonSchema:
    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We convert the tokenizer during set up as this only
        needs to be done once for a given model.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

    def time_lfe(self, _, json_schema_name):
        json_string = json_cases[json_schema_name]["schema"]
        json_samples = json_cases[json_schema_name]["samples"]

        parser = JsonSchemaParser(json_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for json_sample in json_samples:
            json_sample_tokens = self.tokenizer.encode(json_sample)
            for i in range(len(json_sample_tokens)):
                _ = token_enforcer.get_allowed_tokens(json_sample_tokens[: i + 1])
