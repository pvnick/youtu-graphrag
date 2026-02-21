#!/usr/bin/env python3
"""
NTSB Community Profile Analysis
The graph's 100 detected communities cluster accidents by structural similarity
across entity relationships. This script characterizes each community by querying
the graph's community-aware retriever.

Community = a set of accident reports that share dense relational structure
(same aircraft types, same cause factors, same locations, same outcome patterns).
These are NOT predefined categories — they emerge from the graph topology.
"""
import json, time, requests, sys

BASE = "http://localhost:8000"
DATASET = "ntsb_full"
OUT = "/tmp/ntsb_community_profiles.json"
TIMEOUT = 360

# We'll probe the community structure by asking the graph questions that
# require community-level reasoning. The agentic decomposer uses community
# summaries (super nodes) to route sub-questions.
COMMUNITY_PROBE_QUERIES = [
    {
        "id": "comm_cardiac_incapacitation",
        "question": (
            "Describe all accidents in this dataset where pilot incapacitation due to a medical event "
            "(heart attack, stroke, seizure, carbon monoxide poisoning, hypoxia) was a confirmed or "
            "probable cause. What are the common characteristics — pilot age, aircraft type, flight phase, "
            "certificate type? How many such accidents resulted in fatalities versus survivable outcomes?"
        ),
    },
    {
        "id": "comm_experimental_amateur",
        "question": (
            "What accident patterns are specific to experimental amateur-built aircraft? "
            "How do these accidents differ from certified aircraft accidents in cause, phase of flight, "
            "pilot profile, and outcome? What specific construction or design issues appear in narratives?"
        ),
    },
    {
        "id": "comm_agricultural_operations",
        "question": (
            "Describe accidents involving agricultural aviation operations — crop dusting, aerial "
            "application, banner towing, mustering. What are the characteristic risk factors: "
            "low altitude, wire strikes, terrain, pilot fatigue, chemical exposure, repeated low passes?"
        ),
    },
    {
        "id": "comm_flight_instruction",
        "question": (
            "What patterns appear in accidents during flight instruction — student solo accidents, "
            "dual instruction accidents, check ride accidents? How do instructor-involved accidents "
            "differ from student solo accidents in cause and outcome?"
        ),
    },
    {
        "id": "comm_mountain_high_altitude",
        "question": (
            "Describe the cluster of accidents involving mountainous terrain, high density altitude, "
            "or high-elevation airports. What are the common causal patterns — density altitude "
            "performance miscalculation, mountain wave, CFIT in mountain valleys, "
            "oxygen deprivation? What pilot decisions appear repeatedly?"
        ),
    },
    {
        "id": "comm_water_ditching",
        "question": (
            "What accidents involved ditching or unplanned water landings? What were the scenarios — "
            "engine failure over water, spatial disorientation at night over water, "
            "intentional ditching after fuel exhaustion? How often did occupants survive water landings "
            "versus perish from submersion or impact?"
        ),
    },
    {
        "id": "comm_multi_engine_failure",
        "question": (
            "In multi-engine aircraft accidents, what specific failure modes appear — single engine "
            "failure after takeoff, VMC rollover, failure to maintain directional control after "
            "engine failure, both engines failing simultaneously? How do pilot responses to "
            "engine failures in multi-engine aircraft differ in outcome?"
        ),
    },
    {
        "id": "comm_carburetor_icing",
        "question": (
            "How often does carburetor icing appear as a cause or contributing factor in engine "
            "failure accidents? What aircraft types and engine configurations are most affected? "
            "What ambient conditions — temperature, humidity — appear in narratives? "
            "How often did the pilot fail to apply carb heat when it could have prevented the accident?"
        ),
    },
    {
        "id": "comm_controlled_airspace_incursion",
        "question": (
            "What accidents involved airspace violations, controlled airspace incursions, "
            "or mid-air collisions? In mid-air collision accidents, what were the common factors: "
            "VFR-on-VFR traffic conflicts, poor visibility conditions, ATC communication failures, "
            "lack of traffic awareness technology?"
        ),
    },
    {
        "id": "comm_overwater_night_disorientation",
        "question": (
            "Describe accidents where pilots experienced spatial disorientation specifically over "
            "water at night or in featureless terrain. What is the typical accident sequence: "
            "how long from entering disorienting conditions to loss of control? "
            "What instruments, if any, did pilots attempt to use? "
            "Are there cases where autopilot prevented or contributed to disorientation?"
        ),
    },
]


def query(q):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/api/ask-question",
            json={"dataset_name": DATASET, "question": q["question"], "mode": "agent"},
            timeout=TIMEOUT)
        elapsed = round(time.time() - t0, 1)
        if r.status_code == 200:
            data = r.json()
            return {**q, "status": "ok", "elapsed": elapsed,
                    "answer": data.get("answer", ""),
                    "triples": data.get("retrieved_triples", [])[:6]}
        return {**q, "status": "http_error", "code": r.status_code,
                "elapsed": round(time.time()-t0,1)}
    except Exception as e:
        return {**q, "status": "exception", "error": str(e),
                "elapsed": round(time.time()-t0,1)}


def main():
    results = []
    done_ids = set()
    try:
        with open(OUT) as f:
            results = json.load(f)
            done_ids = {r["id"] for r in results if r["status"] == "ok"}
            print(f"Resuming: {len(done_ids)} done", flush=True)
    except Exception:
        pass

    remaining = [q for q in COMMUNITY_PROBE_QUERIES if q["id"] not in done_ids]

    for i, q in enumerate(remaining):
        n = len(done_ids) + i + 1
        print(f"\n{'='*72}", flush=True)
        print(f"[{n}/{len(COMMUNITY_PROBE_QUERIES)}] {q['id']}", flush=True)
        print(f"{'='*72}", flush=True)

        result = query(q)
        results.append(result)

        if result["status"] == "ok":
            print(f"✅ {result['elapsed']}s", flush=True)
            print(f"\n{result['answer'][:600]}\n", flush=True)
        else:
            print(f"❌ {result}", flush=True)

        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(4)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone — {ok}/{len(results)} ok — Results: {OUT}")


if __name__ == "__main__":
    main()
