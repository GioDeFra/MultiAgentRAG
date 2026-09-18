"""Independent RAG evaluation without modifying existing chats or memory.

Install in the project environment:
    pip install "ragas==0.3.2" "langchain-openai>=0.3,<0.4" "sentence-transformers>=3,<6"

Pass a JSON file containing a list of
objects {"question": "...", "reference": "..."} using --input.
    python ragas_evaluation.py --input questions.json
    python ragas_evaluation.py --score-only path/to/dataset.json

Without --score-only, only collect data; Ragas is needed only for scoring.

Each question is independent: no short-term or long-term memory between cases.

Metrics: context precision with reference, context recall, faithfulness,
answer relevancy, and answer correctness.
Answer correctness uses Ragas defaults: 75% factuality and 25% semantic similarity.
DeepSeek handles all LLM evaluations. Answer relevancy and correctness use
local BGE-M3 embeddings; no embedding API or additional API key is used.
The embedding model must already be cached locally. CUDA is used when available,
otherwise embeddings run on CPU. Set EMBEDDINGS["device"] to override this.

All scores are per question, using the final response and combined contexts.

Context precision uses the saved order: selected-agent routing order,
then each agent's internal ranking. This does not represent global reranking.

"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


EVALUATOR = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
}

EMBEDDINGS = {
    "provider": "local",
    "model": "BAAI/bge-m3",
    "device": "auto",
}


def read_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "The JSON file must contain a non-empty list of question/reference objects."
        )
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("question", "reference")
        ):
            raise ValueError(
                f"Case {index}: question and reference must be non-empty strings."
            )
    return cases


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class NoLongTermMemory:
    def __init__(self, *args, **kwargs):
        pass

    def recall_similar(self, *args, **kwargs):
        return []


def collect(cases: list[dict], output: Path) -> list[dict]:
    import agents
    from memory.chat_history import _answer_without_source_appendix

    # Limit the patch to instance construction within this CLI process.
    with patch.object(agents, "LongTermMemory", NoLongTermMemory):
        supervisor = agents.SupervisorAgent(
            chat_history_db_path=str(output / "history.db")
        )
    supervisor._store_in_ltm_background = lambda **kwargs: None
    captured = {}
    for agent_id, agent in supervisor._agents.items():
        original = agent.answer

        def capture(*args, _original=original, _id=agent_id, **kwargs):
            partial = _original(*args, **kwargs)
            captured[_id] = partial
            return partial

        agent.answer = capture

    rows = []
    for index, case in enumerate(cases, 1):
        supervisor.new_session()
        captured.clear()
        row = {
            "id": index,
            "user_input": case["question"],
            "reference": case["reference"],
        }
        print(f"\nCollecting case {index}/{len(cases)}")
        try:
            answer = supervisor.ask(case["question"])
            contexts_by_agent = {
                aid: [
                    f"[{label}] {text}\nSource: {source}"
                    for label, text, _, source in captured[aid].labeled_chunks
                ]
                for aid in supervisor._agents
                if aid in captured
            }
            contexts = [
                text for chunks in contexts_by_agent.values() for text in chunks
            ]
            row.update(
                response=_answer_without_source_appendix(answer),
                retrieved_contexts=contexts,
                contexts_by_agent=contexts_by_agent,
                status="collected" if contexts else "no_context",
            )
        except Exception as exc:
            row.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        # Save a checkpoint before scoring so the RAG does not need to run again.
        write_json(output / "dataset.json", rows)
    return rows


def make_judge():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / "Apikey.env")
    for field in ("base_url", "model", "api_key_env"):
        if not isinstance(EVALUATOR.get(field), str) or not EVALUATOR[field].strip():
            raise ValueError(f"Set EVALUATOR[{field!r}] in ragas_evaluation.py.")
    model = EVALUATOR["model"].strip()
    base_url = EVALUATOR["base_url"].strip()
    key_name = EVALUATOR["api_key_env"].strip()
    api_key = os.getenv(key_name)
    if not api_key or not api_key.strip():
        raise ValueError(
            f"Missing {key_name}: add it to the environment or Apikey.env."
        )

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=180,
        max_retries=2,
    )
    return (
        LangchainLLMWrapper(llm, run_config=RunConfig(timeout=180, max_retries=2)),
        model,
        base_url,
    )


def make_embeddings():
    import torch
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    device = EMBEDDINGS["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Local embeddings: {EMBEDDINGS['model']} on {device}", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS["model"],
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )
    return LangchainEmbeddingsWrapper(embeddings)


async def _score_row(row, metrics, context_metrics, SingleTurnSample):
    """Score one case, retaining skipped metrics and individual errors."""
    result = {"id": row["id"], "question": row["user_input"], "status": row["status"]}
    errors = {}
    if row["status"] == "generation_error":
        errors["generation"] = row.get("error", "Generation failed")
    else:
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )
        for name, metric in metrics.items():
            if name in context_metrics and not row["retrieved_contexts"]:
                result[name] = None
                continue
            try:
                value = float(await metric.single_turn_ascore(sample))
                if not math.isfinite(value):
                    raise ValueError("Score is not finite")
                result[name] = value
            except Exception as exc:
                result[name] = None
                errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"Case {row['id']}: {name} = {result[name]}", flush=True)
    result["errors"] = errors
    return result, errors


async def score(rows: list[dict], output: Path) -> None:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
    )
    from ragas.run_config import RunConfig

    judge, model, base_url = make_judge()
    embeddings = make_embeddings()
    metrics = {
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "answer_correctness": AnswerCorrectness(llm=judge, embeddings=embeddings),
    }
    run_config = RunConfig(timeout=180, max_retries=2)
    for metric in metrics.values():
        metric.init(run_config)
    context_metrics = {"context_precision", "context_recall", "faithfulness"}
    results = []
    for row in rows:
        result, errors = await _score_row(
            row, metrics, context_metrics, SingleTurnSample
        )
        results.append(result)
        write_json(output / "scores.json", results)
        print(f"Evaluated case {row['id']} ({len(errors)} errors)")

    names = list(metrics)
    summary = {
        "judge_model": model,
        "judge_base_url": base_url,
        "total_cases": len(rows),
        "embedding_model": EMBEDDINGS["model"],
        "embedding_provider": EMBEDDINGS["provider"],
        "answer_correctness_weights": metrics["answer_correctness"].weights,
        "answer_relevancy_strictness": metrics["answer_relevancy"].strictness,
        "no_context_cases": sum(r["status"] == "no_context" for r in rows),
        "generation_errors": sum(r["status"] == "generation_error" for r in rows),
        "cases_with_errors": sum(bool(r["errors"]) for r in results),
        "metrics": {},
    }
    for name in names:
        values = [r[name] for r in results if r.get(name) is not None]
        summary["metrics"][name] = {
            "mean": sum(values) / len(values) if values else None,
            "valid_cases": len(values),
        }
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "question", "status", *names, "errors"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {**result, "errors": json.dumps(result["errors"], ensure_ascii=False)}
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", type=Path, help="JSON file containing question and reference fields"
    )
    group.add_argument("--score-only", type=Path, help="Previously collected dataset")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect data only (default behavior)",
    )
    parser.add_argument(
        "--output", type=Path, help="New results directory (must not already exist)"
    )
    args = parser.parse_args()
    if args.score_only and args.collect_only:
        parser.error("--score-only and --collect-only cannot be used together")
    try:
        cases = None if args.score_only else read_cases(args.input)
        rows = (
            json.loads(args.score_only.read_text(encoding="utf-8"))
            if args.score_only
            else None
        )
        if rows is not None and (not isinstance(rows, list) or not rows):
            raise ValueError("Empty or invalid dataset")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = args.output or Path(
        __file__
    ).parent / "ragas_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    if rows is None:
        rows = collect(cases, output)
    else:
        write_json(output / "dataset.json", rows)
    if args.score_only:
        asyncio.run(score(rows, output))
    print(f"\nResults: {output.resolve()}")


if __name__ == "__main__":
    main()