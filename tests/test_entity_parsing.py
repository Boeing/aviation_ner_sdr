"""Regression tests for entity parsing without loading model weights."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class EntityParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_dir = Path(__file__).resolve().parents[1] / "src" / "aviation_ner_sdr"
        spec = importlib.util.spec_from_file_location(
            "run_newnerzero_under_test", source_dir / "run_newnerzero.py"
        )
        module = importlib.util.module_from_spec(spec)
        gliner_stub = types.ModuleType("gliner")
        gliner_stub.GLiNER = object
        with patch.dict(sys.modules, {"gliner": gliner_stub}):
            with patch.object(sys, "path", [str(source_dir)] + sys.path):
                spec.loader.exec_module(module)
        cls.tagger_class = module.NERTagging

    def setUp(self):
        # Parsing does not use the model; bypass its expensive constructor.
        self.tagger = self.tagger_class.__new__(self.tagger_class)

    def test_begin_tags_without_global_labeler(self):
        self.assertEqual(
            self.tagger.parse_out_labels_to_dict(
                [("engine", "b-prod"), ("failure", "b-pcon")]
            ),
            {"prod": ["engine"], "pcon": ["failure"]},
        )

    def test_leading_inside_tag_without_global_labeler(self):
        self.assertEqual(
            self.tagger.parse_out_labels_to_dict([("engine", "i-prod")]),
            {"prod": ["engine"]},
        )

    def test_multi_token_entity_terminated_by_outside_tag(self):
        self.assertEqual(
            self.tagger.parse_out_labels_to_dict(
                [("fuel", "b-prod"), ("pump", "i-prod"), ("failed", "O")]
            ),
            {"prod": ["fuel pump"]},
        )

    def test_empty_input(self):
        self.assertEqual(self.tagger.parse_out_labels_to_dict([]), {})


if __name__ == "__main__":
    unittest.main()
