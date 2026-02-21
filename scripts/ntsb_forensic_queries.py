#!/usr/bin/env python3
"""
NTSB Forensic Analysis — Novel Pattern Mining
Queries designed to surface findings NOT in any existing NTSB publication:
 - Causal chain interior nodes (what happens BETWEEN trigger and crash)
 - Factor interaction effects (co-occurring risks)
 - Hidden contributing factors (in narrative text, not NTSB primary codes)
 - Survivability predictors beyond the obvious
 - Regulatory violation → fatality correlations
 - Pilot decision sequences
 - Latent risk clusters
"""
import json, time, requests, sys

BASE = "http://localhost:8000"
DATASET = "ntsb_full"
OUT = "/tmp/ntsb_forensic_results.json"
TIMEOUT = 360

QUERIES = [

    # ── CAUSAL CHAIN INTERIOR NODES ─────────────────────────────────────────
    # The NTSB codes the FINAL link. The graph encodes the whole chain.
    {
        "id": "chain_loci_interior",
        "theme": "Causal Chains",
        "question": (
            "In accidents where the final outcome was loss of control in flight and crash, "
            "what intermediate events or pilot decisions occurred BETWEEN the initial triggering event "
            "and the actual loss of control? For example: initial engine roughness → pilot diverts attention → "
            "airspeed bleeds off → stall. What are the most common intermediate steps in LOC-I causal chains?"
        ),
    },
    {
        "id": "chain_engine_failure_interior",
        "theme": "Causal Chains",
        "question": (
            "In accidents where engine failure caused a forced landing or crash, trace the events "
            "BEFORE the engine actually failed. What precursor conditions — deferred maintenance, "
            "fuel contamination, oil loss, carburetor ice, pilot-induced fuel exhaustion — most commonly "
            "appear in the narrative before the engine stops? What is the typical gap between the first "
            "warning sign and actual engine failure?"
        ),
    },
    {
        "id": "chain_weather_decision_sequence",
        "theme": "Causal Chains",
        "question": (
            "In weather-related accidents, what sequence of go/no-go decisions did pilots make "
            "before the accident? For example: pilot checks weather → decides to depart despite marginal "
            "conditions → continues into deteriorating weather → encounters IMC → loses control. "
            "What specific decision points appear most often where a different choice would have "
            "broken the chain?"
        ),
    },

    # ── FACTOR INTERACTION EFFECTS ───────────────────────────────────────────
    # Individual risk factors are known. COMBINATIONS are not well-quantified.
    {
        "id": "interaction_night_vfr_mountainous",
        "theme": "Interaction Effects",
        "question": (
            "How do accidents differ when multiple compounding risk factors are present simultaneously: "
            "night conditions, VFR-only pilot, mountainous or elevated terrain, and personal/pleasure flight? "
            "Describe specific accident patterns where 3 or more of these factors co-occurred. "
            "Are these accidents qualitatively different from single-factor accidents?"
        ),
    },
    {
        "id": "interaction_inexperience_complex_aircraft",
        "theme": "Interaction Effects",
        "question": (
            "What accident patterns emerge when a pilot with low total hours or low hours in type "
            "is flying a high-performance, complex, or retractable-gear aircraft? How do these accidents "
            "differ from low-hour pilots flying simple aircraft or experienced pilots in complex aircraft? "
            "What specific failure modes are unique to this combination?"
        ),
    },
    {
        "id": "interaction_selfimposed_pressure",
        "theme": "Interaction Effects",
        "question": (
            "What evidence appears in accident narratives of self-imposed pressure driving poor pilot "
            "decisions — get-home-itis, schedule pressure, continued VFR flight despite deteriorating "
            "conditions, press-on-itis? How often do narratives describe a pilot who was warned, "
            "received weather data suggesting danger, or had prior opportunities to divert but continued? "
            "What are the outcomes when this pressure factor is present?"
        ),
    },

    # ── HIDDEN CONTRIBUTING FACTORS ──────────────────────────────────────────
    # Things buried in narrative text, not in NTSB coded primary cause fields
    {
        "id": "hidden_fatigue_drugs",
        "theme": "Hidden Factors",
        "question": (
            "How often do accident narratives mention pilot fatigue, sleep deprivation, alcohol, "
            "prescription drugs, over-the-counter medication (antihistamines, sedatives), or "
            "marijuana as factors — even when not listed as the primary probable cause? "
            "What substances appear most frequently and in what accident contexts? "
            "Are there patterns in time-of-day, flight duration, or pilot age?"
        ),
    },
    {
        "id": "hidden_distractions_cockpit",
        "theme": "Hidden Factors",
        "question": (
            "What in-cockpit distractions appear in accident narratives — passenger interference, "
            "cellphone use, GPS programming in flight, radio workload, avionics confusion? "
            "How often does distraction appear as a contributing factor even when the probable cause "
            "is listed as something else like loss of control or fuel mismanagement? "
            "What phases of flight are most affected?"
        ),
    },
    {
        "id": "hidden_recency_currency",
        "theme": "Hidden Factors",
        "question": (
            "What patterns exist around pilot recency — how long since last flight, whether the pilot "
            "was current on instrument approach procedures, night currency, or specific aircraft type? "
            "Do narratives reveal pilots who were legally current but operationally rusty? "
            "How often does recency appear as an unspoken factor even when not cited in probable cause?"
        ),
    },

    # ── SURVIVABILITY PREDICTORS ─────────────────────────────────────────────
    # What made some crashes survivable that 'shouldn't' have been?
    {
        "id": "survivability_unexpected",
        "theme": "Survivability",
        "question": (
            "What factors appear in accidents where occupants survived impacts that would typically "
            "be fatal — high-speed terrain impacts, in-flight breakup, post-crash fire, or water impact? "
            "Are there specific aircraft features (parachutes, airframe design, seatbelt type, "
            "crashworthy seats), terrain characteristics, or pilot actions during the crash sequence "
            "that contributed to unexpected survivability?"
        ),
    },
    {
        "id": "survivability_postcrash_fire",
        "theme": "Survivability",
        "question": (
            "How often do accident narratives mention post-crash fire as a cause of death or injury "
            "when the initial impact was survivable? What factors — fuel spillage, electrical systems, "
            "aircraft type, rescue response time — appear in cases where post-crash fire increased casualties? "
            "Are there cases where specific aircraft design or pilot action prevented a fire-related fatality?"
        ),
    },

    # ── REGULATORY VIOLATION × FATALITY ─────────────────────────────────────
    # Which specific violations are strongest predictors of fatal outcomes?
    {
        "id": "regulatory_violations_fatal_correlation",
        "theme": "Regulatory Patterns",
        "question": (
            "What specific regulatory violations appear in accident reports — operating beyond "
            "aircraft limitations, flying without a valid medical, VFR into IMC without instrument "
            "rating, exceeding weight and balance limits, unauthorized flight into controlled airspace, "
            "flying without a current inspection? Which violations most strongly correlate with "
            "fatal outcomes versus non-fatal accidents?"
        ),
    },

    # ── LATENT RISK PROFILES ─────────────────────────────────────────────────
    # Cross-cutting combinations that define underidentified risk populations
    {
        "id": "latent_weekend_warrior_profile",
        "theme": "Latent Risk Clusters",
        "question": (
            "Across accident narratives, describe the profile of accidents involving recreational "
            "pilots on personal flights who fly infrequently (low recent flight hours). "
            "What specific risk factors — aircraft type, weather decisions, time of day, "
            "flight purpose, departure airport type — cluster together in this population? "
            "How do their accidents differ in character from professional or frequent flyers?"
        ),
    },
    {
        "id": "latent_ferry_positioning_flights",
        "theme": "Latent Risk Clusters",
        "question": (
            "What accident patterns appear in ferry flights, positioning flights, or flights to "
            "bring an aircraft to maintenance? Are these flights associated with specific risk factors "
            "like flying aircraft with known issues, unfamiliar aircraft, minimum fuel planning, "
            "or marginal weather acceptance?"
        ),
    },
    {
        "id": "latent_dual_failure_cascade",
        "theme": "Latent Risk Clusters",
        "question": (
            "How often do accident narratives describe a cascade where a first manageable problem "
            "(minor engine roughness, a distraction, a navigation error) led to a second, fatal problem "
            "because the pilot's attention or energy was consumed? Describe the structure of "
            "'dual failure cascade' accidents — what is the typical first failure, how does it consume "
            "the pilot, and what is the second failure that causes the crash?"
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
            return {**q, "status": "ok", "elapsed": elapsed,
                    "answer": r.json().get("answer", ""),
                    "triples": r.json().get("retrieved_triples", [])[:8]}
        return {**q, "status": "http_error", "code": r.status_code,
                "elapsed": round(time.time()-t0,1)}
    except Exception as e:
        return {**q, "status": "exception", "error": str(e),
                "elapsed": round(time.time()-t0,1)}


def main():
    # Resume if partial results exist
    results = []
    done_ids = set()
    try:
        with open(OUT) as f:
            results = json.load(f)
            done_ids = {r["id"] for r in results if r["status"] == "ok"}
            print(f"Resuming: {len(done_ids)} done")
    except Exception:
        pass

    remaining = [q for q in QUERIES if q["id"] not in done_ids]
    total = len(QUERIES)

    for i, q in enumerate(remaining):
        completed = len(done_ids) + i
        print(f"\n{'='*72}", flush=True)
        print(f"[{completed+1}/{total}] [{q['theme']}] {q['id']}", flush=True)
        print(f"{'='*72}", flush=True)

        result = query(q)
        results.append(result)

        if result["status"] == "ok":
            print(f"✅ {result['elapsed']}s", flush=True)
            print(f"\n{result['answer'][:800]}\n", flush=True)
        else:
            print(f"❌ {result}", flush=True)

        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(4)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n\nDONE — {ok}/{len(results)} successful")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
