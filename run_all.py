"""Resumable pipeline: generation -> guard (input/output) -> judge.

Usage:  python run_all.py [stage ...]      stages: info gen guard_in guard_out judge judge2
Every stage appends to results/*.jsonl and skips ids that are already done, so the
script can be stopped and restarted at any time.
"""
import os
import sys
import time

from common import (GUARD, JUDGE, MAX_TOKENS, NO_THINK_RAW, RESULTS, ROOT, TARGETS, THINKING, append_jsonl,
                    chat, generate_raw, keep_awake, llama_guard_prompt, log, parse_guard, read_jsonl, slug,
                    unload_all)

PROMPTS = read_jsonl(os.path.join(ROOT, "prompts.jsonl"))
BY_ID = {p["id"]: p for p in PROMPTS}


def done_ids(path):
    return {r["id"] for r in read_jsonl(path)}


def visible_text(g):
    """What the user sees: the final answer; if generation stopped inside the reasoning trace,
    the (truncated) reasoning is all that was produced, so it is used instead."""
    c = (g.get("content") or "").strip()
    return c if c else (g.get("thinking") or "").strip()


def stage_info():
    import json
    import urllib.request
    from common import OLLAMA
    info = {}
    with urllib.request.urlopen(OLLAMA + "/api/version") as r:
        info["version"] = json.loads(r.read())
    with urllib.request.urlopen(OLLAMA + "/api/tags") as r:
        info["models"] = [{"name": m["name"], "digest": m["digest"], "size": m["size"],
                           "details": m.get("details")} for m in json.loads(r.read())["models"]]
    info["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(RESULTS, "models.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log(f"info: {info['version']} {[m['name'] for m in info['models']]}")


def stage_gen(models=None):
    for model in models or TARGETS:
        unload_all()
        path = os.path.join(RESULTS, f"gen_{slug(model)}.jsonl")
        done = done_ids(path)
        todo = [p for p in PROMPTS if p["id"] not in done]
        log(f"gen {model}: {len(done)} done, {len(todo)} to do")
        t0 = time.time()
        for k, p in enumerate(todo, 1):
            if model in NO_THINK_RAW:
                r = generate_raw(model, NO_THINK_RAW[model].format(prompt=p["prompt"]), MAX_TOKENS[model])
            else:
                think = True if model in THINKING else None
                r = chat(model, [{"role": "user", "content": p["prompt"]}],
                         num_predict=MAX_TOKENS[model], think=think)
            msg = r.get("message", {})
            append_jsonl(path, {
                "id": p["id"], "model": model,
                "content": msg.get("content", ""), "thinking": msg.get("thinking", ""),
                "done_reason": r.get("done_reason"), "eval_count": r.get("eval_count"),
                "prompt_eval_count": r.get("prompt_eval_count"),
                "total_duration_ms": round(r.get("total_duration", 0) / 1e6, 1),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            if k % 25 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                log(f"gen {model}: {k}/{len(todo)}  {rate:.1f}s/item  eta {rate * (len(todo) - k) / 3600:.2f}h")


def classify(model, messages):
    """Llama Guard verdict. User-only turns use the Ollama template (correct for this case);
    User+Agent turns use the official prompt built in common.llama_guard_prompt."""
    t = time.time()
    if len(messages) == 1:
        r = chat(model, messages, num_predict=20)
    else:
        r = generate_raw(model, llama_guard_prompt(model, messages[0]["content"], messages[1]["content"]), 20)
    flagged, cats = parse_guard(r.get("message", {}).get("content", ""))
    return {"flagged": flagged, "cats": cats, "raw": r.get("message", {}).get("content", ""),
            "total_duration_ms": round(r.get("total_duration", 0) / 1e6, 1),
            "wall_ms": round((time.time() - t) * 1000, 1)}


def stage_guard_in():
    path = os.path.join(RESULTS, "guard_input.jsonl")
    done = done_ids(path)
    todo = [p for p in PROMPTS if p["id"] not in done]
    log(f"guard_in: {len(done)} done, {len(todo)} to do")
    for k, p in enumerate(todo, 1):
        res = classify(GUARD, [{"role": "user", "content": p["prompt"]}])
        append_jsonl(path, {"id": p["id"], **res})
        if k % 100 == 0:
            log(f"guard_in: {k}/{len(todo)}")


def stage_guard_out(models=None):
    for model in models or TARGETS:
        gens = read_jsonl(os.path.join(RESULTS, f"gen_{slug(model)}.jsonl"))
        path = os.path.join(RESULTS, f"guard_output_{slug(model)}.jsonl")
        done = done_ids(path)
        # As in Algorithm 1, the output PEP is only reached when the input PEP let the prompt through.
        blocked_in = {r["id"] for r in read_jsonl(os.path.join(RESULTS, "guard_input.jsonl")) if r["flagged"]}
        todo = [g for g in gens if g["id"] not in done and g["id"] not in blocked_in]
        log(f"guard_out {model}: {len(done)} done, {len(todo)} to do")
        for k, g in enumerate(todo, 1):
            p = BY_ID[g["id"]]
            res = classify(GUARD, [{"role": "user", "content": p["prompt"]},
                                   {"role": "assistant", "content": visible_text(g) or "(no output)"}])
            append_jsonl(path, {"id": g["id"], "model": model, **res})
            if k % 100 == 0:
                log(f"guard_out {model}: {k}/{len(todo)}")


def stage_judge(models=None, judge_model=JUDGE, tag="judge"):
    """Offline evaluator: did the response fulfil the ORIGINAL harmful goal in an unsafe way?"""
    for model in models or TARGETS:
        gens = read_jsonl(os.path.join(RESULTS, f"gen_{slug(model)}.jsonl"))
        path = os.path.join(RESULTS, f"{tag}_{slug(model)}.jsonl")
        done = done_ids(path)
        # Only responses to harmful requests are judged (benign ones are scored for over-refusal).
        todo = [g for g in gens if g["id"] not in done and BY_ID[g["id"]]["set"] == "harmful"]
        log(f"{tag} {model}: {len(done)} done, {len(todo)} to do")
        t0 = time.time()
        for k, g in enumerate(todo, 1):
            p = BY_ID[g["id"]]
            res = classify(judge_model, [{"role": "user", "content": p["goal"]},
                                         {"role": "assistant", "content": visible_text(g) or "(no output)"}])
            append_jsonl(path, {"id": g["id"], "model": model, **res})
            if k % 50 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                log(f"{tag} {model}: {k}/{len(todo)}  {rate:.1f}s/item  eta {rate * (len(todo) - k) / 3600:.2f}h")


STAGES = {"info": stage_info, "gen": stage_gen, "guard_in": stage_guard_in,
          "guard_out": stage_guard_out, "judge": stage_judge}

def run_per_model():
    """Default order: finish every stage for one model before moving to the next one,
    so that complete results are available as early as possible."""
    stage_info()
    unload_all()
    stage_guard_in()
    for model in TARGETS:
        for fn in (stage_gen, stage_guard_out, stage_judge):
            unload_all()
            fn([model])


if __name__ == "__main__":
    keep_awake()
    stages = sys.argv[1:]
    log(f"=== run_all start: {stages or 'per-model'}")
    try:
        if not stages:
            run_per_model()
        for s in stages:
            unload_all()
            STAGES[s]()
        log("=== run_all finished OK")
    except Exception as ex:
        import traceback
        log(f"=== run_all FAILED: {ex}\n{traceback.format_exc()}")
        raise
