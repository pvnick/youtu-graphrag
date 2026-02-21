"""
add_aggregates.py — Add aggregate count/summary nodes to the graph.

Problem: "Which aircraft is involved in most accidents?" fails because the graph
stores individual accident→aircraft edges but has no aggregated count nodes.

This script adds:
  - aircraft_type → [accident_count: N] → summary_node
  - cause_factor → [accident_count: N] → summary_node
  - phase_of_flight → [accident_count: N] → summary_node
  - injury_level → [accident_count: N] → summary_node
  - location → [accident_count: N] → summary_node (top 20 states/regions)

These aggregate nodes allow queries like "What aircraft type has the most
accidents?" to be answered from graph traversal.

Usage:
    cd /root/youtu-graphrag
    python scripts/add_aggregates.py --dataset ntsb_sample
"""
import argparse, json, os, sys, collections

sys.path.insert(0, '/root/youtu-graphrag')
from config import get_config

# How many top entities to add aggregates for (to keep graph manageable)
TOP_K = 50

AGGREGATE_CONFIGS = [
    {
        'relation': 'involved_aircraft',
        'target_type': 'aircraft',
        'agg_relation': 'has_accident_count',
        'label': 'aircraft_type',
        'description': 'aircraft type involvement in accidents',
    },
    {
        'relation': 'caused_by',
        'target_type': 'cause_factor',
        'agg_relation': 'has_accident_count',
        'label': 'cause_factor',
        'description': 'cause factor frequency across accidents',
    },
    {
        'relation': 'resulted_in',
        'target_type': 'cause_factor',
        'agg_relation': 'has_accident_count',
        'label': 'cause_factor',
        'description': 'cause factor outcome frequency',
    },
    {
        'relation': None,  # any relation to phase_of_flight
        'target_type': 'phase_of_flight',
        'agg_relation': 'has_accident_count',
        'label': 'phase_of_flight',
        'description': 'accidents by phase of flight',
    },
    {
        'relation': None,
        'target_type': 'injury_outcome',
        'agg_relation': 'has_accident_count',
        'label': 'injury_outcome',
        'description': 'accidents by injury severity',
    },
    {
        'relation': None,
        'target_type': 'maintenance_issue',
        'agg_relation': 'has_accident_count',
        'label': 'maintenance_issue',
        'description': 'accidents by maintenance issue type',
    },
]


def count_by_entity(edges, target_type, relation=None):
    """Count how many unique accident chunks reference each entity of the given type."""
    entity_chunks = collections.defaultdict(set)

    for e in edges:
        for side, other_side in [('start_node', 'end_node'), ('end_node', 'start_node')]:
            node = e[side]
            other = e[other_side]
            props = node.get('properties', {})
            other_props = other.get('properties', {})

            if props.get('schema_type') == target_type:
                if relation is None or e['relation'] == relation:
                    # Count by chunk ID (= unique accident)
                    chunk_id = props.get('chunk id') or other_props.get('chunk id', 'unknown')
                    entity_name = props.get('name', '')
                    if entity_name:
                        entity_chunks[entity_name].add(chunk_id)

    # Return sorted (name, count) pairs
    counts = [(name, len(chunks)) for name, chunks in entity_chunks.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts


def make_aggregate_node(entity_name, count, schema_type, description):
    """Create a summary aggregate node."""
    return {
        'label': 'entity',
        'properties': {
            'name': f'{entity_name} [aggregate]',
            'schema_type': f'{schema_type}_aggregate',
            'accident_count': count,
            'description': description,
            'is_aggregate': True,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    args = parser.parse_args()

    config = get_config()
    ds_config = config.get_dataset_config(args.dataset)
    graph_path = ds_config.graph_output
    if not os.path.exists(graph_path):
        for alt in [graph_path.replace(".json","_new.json"), f"output/graphs/{args.dataset}_new.json"]:
            if os.path.exists(alt): graph_path = alt; break

    print(f"Loading graph: {graph_path}")
    with open(graph_path) as f:
        edges = json.load(f)
    print(f"  {len(edges):,} edges")

    new_edges = list(edges)
    total_added = 0

    for cfg in AGGREGATE_CONFIGS:
        target_type = cfg['target_type']
        relation = cfg.get('relation')
        agg_relation = cfg['agg_relation']
        description = cfg['description']

        print(f"\nCounting '{target_type}' entities (relation={relation or 'any'}) ...")
        counts = count_by_entity(edges, target_type, relation)

        top = counts[:args.top_k]
        print(f"  Top {len(top)} by accident count:")
        for name, count in top[:10]:
            print(f"    {count:4d}x  {name[:60]}")
        if len(top) > 10:
            print(f"    ... and {len(top)-10} more")

        added = 0
        for entity_name, count in top:
            if count < 2:
                continue  # skip singletons

            # Source node: the entity itself
            source_node = {
                'label': 'entity',
                'properties': {
                    'name': entity_name,
                    'schema_type': target_type,
                }
            }

            # Target: aggregate summary node
            agg_node = make_aggregate_node(entity_name, count, target_type, description)

            # Edge: entity → [has_accident_count] → aggregate
            new_edges.append({
                'start_node': source_node,
                'relation': agg_relation,
                'end_node': agg_node,
            })

            # Also add a global ranking edge to a "STATISTICS" node
            stats_node = {
                'label': 'entity',
                'properties': {
                    'name': f'NTSB Statistics: {target_type}',
                    'schema_type': 'statistics',
                    'description': f'Aggregate statistics for {target_type} across all accidents',
                }
            }
            new_edges.append({
                'start_node': agg_node,
                'relation': 'ranked_in',
                'end_node': stats_node,
            })
            added += 1

        print(f"  Added {added} aggregate nodes")
        total_added += added

    print(f"\nTotal new edges added: {total_added * 2}")
    print(f"Graph size: {len(edges):,} → {len(new_edges):,} edges")

    print(f"Saving graph ...")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(new_edges, f, ensure_ascii=False, indent=2)
    print(f"✅ Aggregate nodes saved to {graph_path}")


if __name__ == '__main__':
    main()
