import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "lib_chromadb.py"


class FakeCollection:
    def __init__(self, threshold, metadata=None):
        self.configuration_json = {"hnsw": {"sync_threshold": threshold}}
        self.metadata = dict(metadata or {})
        self.modifications = []

    def modify(self, *, configuration=None, metadata=None):
        self.modifications.append(
            {"configuration": configuration, "metadata": metadata}
        )
        if configuration is not None:
            self.configuration_json["hnsw"].update(configuration["hnsw"])
        if metadata is not None:
            self.metadata = dict(metadata)


def load_module(collection, threshold="20"):
    class FakeClient:
        def get_or_create_collection(self, **_kwargs):
            return collection

    chromadb = types.SimpleNamespace(PersistentClient=lambda **_kwargs: FakeClient())
    spec = importlib.util.spec_from_file_location("lib_chromadb_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"chromadb": chromadb}), mock.patch.dict(
        os.environ, {"METTACLAW_CHROMA_SYNC_THRESHOLD": threshold}, clear=False
    ):
        spec.loader.exec_module(module)
    return module


class ConfigurationTests(unittest.TestCase):
    def test_existing_collection_is_migrated_and_metadata_is_synchronized(self):
        collection = FakeCollection(
            100,
            {
                "embedding_protocol": "test",
                "hnsw:space": "cosine",
                "hnsw:sync_threshold": 100,
            },
        )

        module = load_module(collection)

        self.assertEqual(module.SYNC_THRESHOLD, 20)
        self.assertEqual(collection.configuration_json["hnsw"]["sync_threshold"], 20)
        self.assertEqual(collection.metadata["hnsw:sync_threshold"], 20)
        self.assertEqual(collection.metadata["embedding_protocol"], "test")
        self.assertNotIn("hnsw:space", collection.metadata)
        self.assertEqual(
            collection.modifications[0]["configuration"],
            {"hnsw": {"sync_threshold": 20}},
        )

    def test_matching_collection_is_not_modified(self):
        collection = FakeCollection(20, {"hnsw:sync_threshold": 20})

        load_module(collection)

        self.assertEqual(collection.modifications, [])

    def test_positive_environment_override_is_used(self):
        collection = FakeCollection(20, {"hnsw:sync_threshold": 20})

        module = load_module(collection, threshold="7")

        self.assertEqual(module.SYNC_THRESHOLD, 7)
        self.assertEqual(collection.configuration_json["hnsw"]["sync_threshold"], 7)

    def test_nonpositive_threshold_is_rejected(self):
        collection = FakeCollection(20, {"hnsw:sync_threshold": 20})

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            load_module(collection, threshold="0")


if __name__ == "__main__":
    unittest.main()
