import json
import os
import random
import tempfile
import unittest

from benchmarks.data_utils.synthesizer import Synthesizer


# Helper function to create and dump data
def dump_record(handle, timestamp, hash_ids, block_size=512):
    input_length = block_size * len(hash_ids)
    output_length = random.randint(50, 250)

    data = {
        "timestamp": timestamp,
        "hash_ids": hash_ids,
        "input_length": input_length,
        "output_length": output_length,
    }
    json.dump(data, handle)
    handle.write("\n")


def test_graph_structure():
    # Create a temporary JSONL file with the specified data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        dump_record(tmp, 1000, [0, 1, 2, 3, 4])
        dump_record(tmp, 2000, [0, 1, 2])

    # Create the Synthesizer with the temporary file
    synthesizer = Synthesizer(tmp.name, block_size=512)

    # Verify the graph structure
    # Check that the root node (-1) has only one child
    root_successors = list(synthesizer.G.successors(-1))
    assert len(root_successors) == 1, "Root node should have exactly one child"

    # Verify that the child is node 2 with length 3
    assert (
        root_successors[0] == 2
    ), f"Root's child should be node 2, but is {root_successors[0]}"
    assert synthesizer.G.nodes[2]["length"] == 3, "Node 2 should have length 3"

    # Verify the edge weight from root to child is 2
    assert (
        synthesizer.G[-1][2]["weight"] == 2
    ), "Edge weight from root to node 2 should be 2"

    # Clean up
    os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
