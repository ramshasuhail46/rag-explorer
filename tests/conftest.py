"""Shared test fixtures. pytest loads this file automatically."""
import pandas as pd
import pytest


@pytest.fixture
def tiny_docs() -> pd.DataFrame:
    """Five short documents with clearly different topics."""
    return pd.DataFrame(
        {
            "id": ["d1", "d2", "d3", "d4", "d5"],
            "title": ["Transformers", "Hallucination", "Image CNNs",
                      "Reinforcement learning", "Speech recognition"],
            "abstract": [
                "Transformers use self-attention to model sequences.",
                "Retrieval reduces hallucination in large language models.",
                "Convolutional networks classify images using filters.",
                "Agents learn policies by maximizing reward signals.",
                "Acoustic models convert speech audio into text.",
            ],
            "categories": ["cs.CL", "cs.CL", "cs.CV", "cs.LG", "cs.SD"],
            "update_date": ["2023-01-01"] * 5,
        }
    )