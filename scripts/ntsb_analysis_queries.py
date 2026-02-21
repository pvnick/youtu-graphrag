#!/usr/bin/env python3
"""
NTSB Full Dataset Analysis — Root Cause Discovery
Runs a structured series of analytical queries against the ntsb_full graph.
Outputs results to /tmp/ntsb_analysis_results.json and a human-readable report.
"""
import json, time, requests, sys

BASE = "http://localhost:8000"
DATASET = "ntsb_full"
OUT_FILE = "/tmp/ntsb_analysis_results.json"
TIMEOUT = 240  # seconds per query

QUERIES = [
    # ── Category 1: Aggregate root causes ────────────────────────────────────
    {
        "id": "root_causes_top",
        "category": "Root Causes",
        "question": "What are the top 5 most common root causes and contributing factors across all aviation accidents in this dataset? Provide specific cause categories and estimate their relative frequency.",
    },
    {
        "id": "pilot_error_breakdown",
        "category": "Human Factors",
        "question": "Break down pilot error accidents into subcategories. What specific pilot errors are most common — poor decision-making, loss of control, inadequate preflight planning, spatial disorientation, fuel mismanagement? Which are most deadly?",
    },
    {
        "id": "weather_accidents",
        "category": "Environmental",
        "question": "What weather conditions are most associated with aviation accidents? How do VFR-into-IMC accidents differ in outcome from other weather-related accidents? What specific weather factors are most lethal?",
    },
    {
        "id": "mechanical_failures",
        "category": "Mechanical",
        "question": "What are the most common mechanical failures and aircraft system issues leading to accidents? Which specific aircraft systems (engine, fuel, flight controls, landing gear) fail most often and which failures are most fatal?",
    },
    {
        "id": "phase_of_flight",
        "category": "Operational",
        "question": "Which phases of flight have the highest accident rates — takeoff, climb, cruise, approach, landing, maneuvering? For each phase, what are the primary causes and what is the fatality rate?",
    },
    # ── Category 2: Specific risk patterns ───────────────────────────────────
    {
        "id": "loss_of_control",
        "category": "Loss of Control",
        "question": "Analyze loss of control in flight (LOC-I) accidents. What are the common precursors: low altitude maneuvering, stall-spin, VFR into IMC, distraction? What pilot experience profiles appear in these accidents?",
    },
    {
        "id": "fuel_mismanagement",
        "category": "Fuel & Systems",
        "question": "How many accidents involved fuel exhaustion, fuel starvation, or improper fuel management? What are the common failure patterns — failure to check fuel, wrong tank selected, improper fuel planning? Are these accidents preventable?",
    },
    {
        "id": "maintenance_issues",
        "category": "Maintenance",
        "question": "What maintenance-related issues and regulatory violations contributed to accidents? Are there patterns around deferred maintenance, improper repairs, or inspection failures? Which maintenance failures are most dangerous?",
    },
    {
        "id": "pilot_experience",
        "category": "Human Factors",
        "question": "What patterns exist around pilot experience levels in accidents? Are accidents more common among low-hour pilots? What is the relationship between pilot total hours, hours in type, and accident severity?",
    },
    {
        "id": "airport_approach",
        "category": "Approach & Landing",
        "question": "What are the primary causes of approach and landing accidents? How do controlled flight into terrain (CFIT), runway excursions, and unstabilized approaches contribute? What conditions — night, low visibility, unfamiliar airports — increase risk?",
    },
    # ── Category 3: Systemic patterns ─────────────────────────────────────────
    {
        "id": "recurring_cause_chains",
        "category": "Causal Chains",
        "question": "Identify the most common causal chains — sequences of events that lead from an initiating factor to a fatal outcome. For example: VFR into IMC → spatial disorientation → loss of control → impact. What are the top 3 causal chains?",
    },
    {
        "id": "aircraft_type_risk",
        "category": "Aircraft Type",
        "question": "Which aircraft types, makes, or categories appear most frequently in accidents? Are certain aircraft makes or models disproportionately involved in accidents? What are the most common failure modes for high-frequency accident aircraft?",
    },
    {
        "id": "geographic_patterns",
        "category": "Geographic",
        "question": "Are there geographic patterns in aviation accidents? What types of terrain or airport environments are associated with higher accident rates — mountainous terrain, remote areas, high-density airspace?",
    },
    # ── Category 4: Preventability and interventions ─────────────────────────
    {
        "id": "preventable_accidents",
        "category": "Prevention",
        "question": "Based on accident narratives, what proportion of accidents involved clearly preventable pilot decisions? What are the most commonly cited probable cause findings that suggest a specific intervention could have prevented the accident?",
    },
    {
        "id": "fatal_vs_nonfatal",
        "category": "Injury Outcomes",
        "question": "What factors differentiate fatal accidents from non-fatal accidents? Are there specific cause factors, phases of flight, aircraft types, or conditions that are significantly more likely to result in fatalities versus survivable accidents?",
    },
]


def query(q_id, question):
    payload = {
        "dataset_name": DATASET,
        "question": question,
        "mode": "agent"
    }
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/api/ask-question", json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
        if r.status_code == 200:
            d = r.json()
            return {
                "id": q_id,
                "status": "ok",
                "elapsed": round(elapsed, 1),
                "answer": d.get("answer", ""),
                "triples": d.get("retrieved_triples", [])[:10],
                "chunks": d.get("retrieved_chunks", [])[:3],
            }
        else:
            return {"id": q_id, "status": "error", "code": r.status_code, "body": r.text[:200], "elapsed": round(elapsed,1)}
    except Exception as e:
        return {"id": q_id, "status": "exception", "error": str(e), "elapsed": round(time.time()-t0, 1)}


def main():
    results = []
    total = len(QUERIES)

    for i, q in enumerate(QUERIES):
        print(f"\n{'='*70}", flush=True)
        print(f"[{i+1}/{total}] {q['category']}: {q['id']}", flush=True)
        print(f"Q: {q['question'][:120]}...", flush=True)
        print(f"{'='*70}", flush=True)

        result = query(q["id"], q["question"])
        result["category"] = q["category"]
        result["question"] = q["question"]
        results.append(result)

        if result["status"] == "ok":
            print(f"✅ Done in {result['elapsed']}s", flush=True)
            print(f"\n{result['answer'][:600]}...\n", flush=True)
        else:
            print(f"❌ {result}", flush=True)

        # Save incrementally
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        # Brief pause between queries
        time.sleep(5)

    print(f"\n\n{'='*70}", flush=True)
    print(f"ALL DONE — {len([r for r in results if r['status']=='ok'])}/{total} successful", flush=True)
    print(f"Results saved to {OUT_FILE}", flush=True)

    return results


if __name__ == "__main__":
    main()
