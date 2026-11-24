import json
import sys
from types import SimpleNamespace

import pytest

from curate_eval.filters import FilterConfig, _language_matches, _minhash_deduplicate, curate
from curate_eval import ablation
from curate_eval import evaluate
from curate_eval.io import read_examples, write_examples
from curate_eval.judge_validation import cohen_kappa
from curate_eval import judge
from curate_eval.schema import Example
from curate_eval.stackexchange import convert
from curate_eval.wikipedia import convert as convert_wikipedia


def example(instruction="Explain a useful programming concept clearly",
            output="A programming concept describes a reusable idea with a practical example."):
    return Example(instruction=instruction, output=output)


def test_jsonl_roundtrip_preserves_unicode_and_metadata(tmp_path):
    path = tmp_path / "examples.jsonl"
    row = Example(instruction="¿Qué es una API?", output="Una interfaz para programas.")
    write_examples(path, [row])
    loaded = read_examples(path)
    assert loaded[0].instruction == row.instruction


def test_parquet_roundtrip_preserves_heterogeneous_metadata(tmp_path):
    path = tmp_path / "examples.parquet"
    rows = [
        Example(instruction="Explain an API clearly", output="It is a defined interface.",
                metadata={"source": "StackExchange", "question_id": "10"}),
        Example(instruction="Explain a database index", output="It speeds up data lookup.",
                metadata={"synthetic": True, "seed_source_id": "example-1"}),
    ]
    write_examples(path, rows)
    loaded = read_examples(path)
    assert [example.metadata for example in loaded] == [example.metadata for example in rows]


def test_curate_removes_exact_duplicate_and_scrubs_email():
    rows = [example(output="A clear answer with a useful contact hello@example.org."),
            example(output="A clear answer with a useful contact hello@example.org.")]
    result, report = curate(rows, FilterConfig(min_instruction_chars=5, min_output_chars=10))
    assert len(result) == 1
    assert "[EMAIL]" in result[0].output
    assert report["exact_duplicates"] == 1
    assert report["pii_scrubbed"] == 1


def test_decontamination_drops_exact_eval_match(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    write_examples(eval_path, [example()])
    result, report = curate([example()], FilterConfig(
        min_instruction_chars=5, min_output_chars=10, eval_path=str(eval_path)))
    assert result == []
    assert report["eval_contamination"] == 1


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_cohen_kappa_rejects_undefined_expected_agreement():
    with pytest.raises(ValueError, match="undefined"):
        cohen_kappa([1, 1], [1, 1])


def test_call_judge_can_request_json_mode(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    config = FilterConfig(judge_url="https://judge.example/v1/chat/completions", judge_model="test-model")

    assert judge.call_judge("Return JSON", config, json_mode=True) == "{}"
    payload = json.loads(captured["request"].data)
    assert captured["request"].full_url == config.judge_url
    assert payload["response_format"] == {"type": "json_object"}


def test_judge_win_rate_accepts_a_choice_followed_by_explanation(monkeypatch):
    monkeypatch.setattr(evaluate, "call_judge", lambda *args, **kwargs: "A\n\nCandidate A is more complete.")
    config = FilterConfig(judge_url="https://judge.example/v1/chat/completions", judge_model="test-model")

    assert evaluate.judge_win_rate(["A complete prediction"], ["A short reference"], [example()], config) == 1.0


def test_lm_eval_uses_mps_when_cuda_is_unavailable(tmp_path, monkeypatch):
    model_dir = tmp_path / "adapter"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "test/base-model"}),
        encoding="utf-8",
    )
    test_data = tmp_path / "test.jsonl"
    test_data.write_text("", encoding="utf-8")
    commands = []
    monkeypatch.setattr(ablation.shutil, "which", lambda _: "/usr/bin/lm_eval")
    monkeypatch.setattr(ablation.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    ))

    ablation.run_lm_eval(model_dir, str(test_data), tmp_path / "lm_eval.json")

    model_args = commands[0][commands[0].index("--model_args") + 1]
    assert model_args == f"pretrained=test/base-model,peft={model_dir.resolve()},device=mps"


def test_minhash_detects_near_duplicate_above_jaccard_threshold():
    first = example(output="A clear answer explains a reusable programming idea using practical examples.")
    near = example(output="A clear answer explains a reusable programming idea using practical example.")
    assert len(_minhash_deduplicate([first, near], threshold=0.80, permutations=64)) == 1


def test_stackexchange_converter_only_keeps_accepted_answers(tmp_path):
    source = tmp_path / "Posts.xml"
    source.write_text(
        '<posts><row Id="1" PostTypeId="1" AcceptedAnswerId="2" Title="How do APIs work?" '
        'Body="Question body" Tags="&lt;api&gt;"/>'
        '<row Id="2" PostTypeId="2" ParentId="1" Body="Accepted answer with a useful explanation."/>'
        '<row Id="3" PostTypeId="2" ParentId="1" Body="Another answer that is not accepted."/></posts>',
        encoding="utf-8",
    )
    output = tmp_path / "raw.jsonl"
    assert convert(str(source), str(output), "https://stackoverflow.com") == 1
    rows = read_examples(output)
    assert len(rows) == 1
    assert rows[0].source_id == "2"
    assert rows[0].metadata["answer_url"] == "https://stackoverflow.com/a/2"


def test_wikipedia_shard_converter_preserves_source_attribution(tmp_path):
    import gzip

    source = tmp_path / "pcmwiki.txt.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("Nigerian Pidgin\n")
        handle.write("Nigerian Pidgin na language wey dem dey tok for Nigeria.\n\n")
        handle.write("Lagos\n")
        handle.write("Lagos na big city for Nigeria.\n\n")
    output = tmp_path / "pidgin.jsonl"
    assert convert_wikipedia(source, output) == 2
    examples = read_examples(output)
    assert examples[0].instruction == "Wetin be Nigerian Pidgin?"
    assert examples[0].language == "pcm"
    assert examples[0].metadata["source_license"] == "CC BY-SA 3.0"
    assert examples[0].metadata["article_url"].endswith("/Nigerian_Pidgin")


def test_source_language_label_requires_provenance_when_trusted():
    tagged = Example(
        instruction="Wetin be Lagos?",
        output="Lagos na big city for Nigeria.",
        language="pcm",
        metadata={"source_dataset": "corpus", "source_revision": "abc123"},
    )
    assert _language_matches(tagged, "pcm", trust_source_label=True)
    unproven = Example(instruction=tagged.instruction, output=tagged.output, language="pcm")
    with pytest.raises(ValueError, match="source_dataset"):
        _language_matches(unproven, "pcm", trust_source_label=True)


def test_ablation_builds_missing_combined_dataset_from_filtered_seeds(tmp_path, monkeypatch):
    seed = Example(
        instruction="Wetin be Nigerian Pidgin?",
        output="Nigerian Pidgin na language wey dem dey tok for Nigeria.",
        source_id="article-1",
        language="pcm",
        metadata={"source_dataset": "wiki", "source_revision": "rev1"},
    )
    synthetic = Example(
        instruction="How dem dey use Pidgin for Lagos?",
        output="Many pipul for Lagos dey use Nigerian Pidgin for everyday communication.",
        source_id="synthetic-17-1",
        language="pcm",
        metadata={"synthetic": True, "source_dataset": "wiki", "source_revision": "rev1"},
    )
    filtered_path = tmp_path / "filtered.jsonl"
    combined_path = tmp_path / "filtered-plus-synthetic.jsonl"
    candidates_path = tmp_path / "synthetic-candidates.jsonl"
    write_examples(filtered_path, [seed])
    monkeypatch.setattr(ablation, "generate", lambda *args, **kwargs: [synthetic])
    monkeypatch.setattr(ablation, "curate", lambda examples, config: (examples, {"output": len(examples)}))

    report = ablation.prepare_filtered_synthetic(
        str(filtered_path),
        str(combined_path),
        str(candidates_path),
        1,
        "https://judge.example/v1/chat/completions",
        "test-model",
        language="pcm",
        trust_language_labels=True,
    )
    result = read_examples(combined_path)
    assert report["retained_synthetic"] == 1
    assert len(result) == 2
    assert candidates_path.is_file()
    assert (tmp_path / "filtered-plus-synthetic.report.json").is_file()


def test_ablation_uses_identical_training_arguments_for_each_arm(tmp_path, monkeypatch):
    seed = Example(instruction="A useful prompt about computing", output="A sufficiently detailed answer about computing.")
    synthetic = Example(
        instruction="A synthetic prompt about mathematics",
        output="A sufficiently detailed answer about mathematics.",
        source_id="synthetic-1",
        metadata={"synthetic": True},
    )
    raw_path, filtered_path, combined_path = (tmp_path / name for name in (
        "raw.jsonl", "filtered.jsonl", "combined.jsonl"))
    write_examples(raw_path, [seed])
    write_examples(filtered_path, [seed])
    write_examples(combined_path, [seed, synthetic])
    commands = []
    monkeypatch.setattr(ablation.subprocess, "run", lambda command, check: commands.append(command))

    ablation.run(
        {"raw": str(raw_path), "filtered": str(filtered_path), "filtered_synthetic": str(combined_path)},
        "test/base-model",
        str(tmp_path / "runs"),
        seed=23,
        epochs=2,
        learning_rate=1e-4,
        batch_size=3,
        max_length=512,
        max_steps=3,
        gradient_accumulation_steps=1,
    )
    assert len(commands) == 3
    for option in (
        "--seed", "--epochs", "--learning-rate", "--batch-size", "--max-length",
        "--max-steps", "--gradient-accumulation-steps",
    ):
        values = [command[command.index(option) + 1] for command in commands]
        assert len(set(values)) == 1
    assert commands[0][commands[0].index("--learning-rate") + 1] == "0.0001"
    assert commands[0][commands[0].index("--max-steps") + 1] == "3"
    assert commands[0][commands[0].index("--gradient-accumulation-steps") + 1] == "1"
