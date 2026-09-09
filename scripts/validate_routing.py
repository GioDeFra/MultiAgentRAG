"""Live routing regression check; no Pinecone writes, model downloads or chat history writes.

Run from the repository root: python scripts/validate_routing.py
Uses the provider configured in Apikey.env and makes actual LLM calls.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import get_llm_client, model_names
from routing import JurisdictionRouter, country_question


def main():
    router = JurisdictionRouter(get_llm_client().with_options(timeout=90, max_retries=1), model_names()["main"])
    pending = [{"query": "Come funziona il divorzio?", "answer": country_question("it")}]
    italy = [{"query": "Come funziona il divorzio in Italia?", "answer": "Abbiamo discusso del divorzio italiano."}]
    cases = [
        ("missing", "Come funziona il divorzio?", [], set(), set(), True),
        ("one", "Come funziona il divorzio in Italia?", [], {"Italy"}, set(), False),
        ("two", "Confronta il divorzio in Italia e Slovenia", [], {"Italy", "Slovenia"}, set(), False),
        ("three", "Come funziona il divorzio in Italia, Slovenia ed Estonia?", [], {"Italy", "Slovenia", "Estonia"}, set(), False),
        ("external", "Come funziona il divorzio in Francia?", [], set(), {"France"}, False),
        ("mixed", "Confronta il divorzio in Italia e Francia", [], {"Italy"}, {"France"}, False),
        ("reply_one", "Italia", pending, {"Italy"}, set(), False),
        ("reply_two", "Italia e Slovenia", pending, {"Italy", "Slovenia"}, set(), False),
        ("reply_all", "Tutti e tre", pending, {"Italy", "Slovenia", "Estonia"}, set(), False),
        ("reply_external", "Francia", pending, set(), {"France"}, False),
        ("followup", "E per l'affidamento dei figli?", italy, {"Italy"}, set(), False),
        ("options_not_selection", "Quali documenti servono per il divorzio?", pending, set(), set(), True),
        ("ambiguous_two", "Due Paesi", pending, set(), set(), True),
        ("exclude", "Il divorzio in Slovenia, non in Italia", [], {"Slovenia"}, set(), False),
        ("new_topic", "Come funziona l'eredità?", italy, set(), set(), True),
    ]

    def run(case):
        name, question, history, covered, external, clarify = case
        route = router.route(question, history)
        actual = {c for aid in route.agent_ids for c in router.registry[aid].countries}
        passed = actual == covered and set(route.external_countries) == external and route.clarification == clarify
        if name.startswith("reply_"):
            passed = passed and ("divorz" in route.query.lower() or "divorce" in route.query.lower())
        result = {"case": name, "passed": passed, "route": asdict(route)}
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run, cases))
    external = next(item for item in results if item["case"] == "external")
    if external["passed"]:
        from routing import RouteDecision
        answer = router.answer_external(RouteDecision(**external["route"]))
        results.append({"case": "external_answer", "passed": "senza fonti del RAG" in answer and len(answer) > 120,
                        "answer": answer})
    output = Path(__file__).resolve().parents[1] / "ragas_results" / "routing-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Passed {sum(item['passed'] for item in results)}/{len(results)}. Report: {output}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
