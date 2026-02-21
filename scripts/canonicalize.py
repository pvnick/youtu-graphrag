"""
canonicalize.py — Merge near-duplicate entity nodes in the graph.

Problem: "loss of control", "Loss of Control", "LOC-I", "loss of airplane control"
are separate disconnected nodes, fragmenting the graph and hurting retrieval.

Approach:
  1. Group entities by schema_type
  2. Embed all entity names within each type
  3. Build a FAISS index and find pairs with cosine sim > threshold
  4. Union-Find to build merge groups
  5. Rewrite graph edges, replacing merged nodes with the canonical node
     (canonical = the highest-degree node in the merge group)

Usage:
    cd /root/youtu-graphrag
    python scripts/canonicalize.py --dataset ntsb_sample [--threshold 0.92] [--dry-run]
"""
import argparse, json, os, sys, time, collections
import numpy as np

sys.path.insert(0, '/root/youtu-graphrag')
from config import get_config

def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_batch(model, texts, batch_size=256):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
    return np.vstack(all_embs).astype('float32')

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

def find_merge_groups(names, embeddings, threshold, type_name):
    """Use FAISS to find near-duplicate entity names."""
    import faiss
    n = len(names)
    if n < 2:
        return {}

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    k = min(10, n)
    D, I = index.search(embeddings, k)

    uf = UnionFind(n)
    merge_count = 0
    for i in range(n):
        for j_idx in range(1, k):
            j = I[i, j_idx]
            sim = D[i, j_idx]
            if sim >= threshold and i != j:
                uf.union(i, j)
                merge_count += 1

    # Group by representative
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    # Only return groups with >1 member
    merge_groups = {rep: members for rep, members in groups.items() if len(members) > 1}
    if merge_groups:
        print(f"    {type_name}: {len(merge_groups)} merge groups from {sum(len(v) for v in merge_groups.values())} entities (threshold={threshold})")
    return merge_groups

def build_degree_map(edges):
    """Count edges per node name for canonical node selection."""
    degree = collections.Counter()
    for e in edges:
        degree[e['start_node']['properties'].get('name', '')] += 1
        degree[e['end_node']['properties'].get('name', '')] += 1
    return degree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--threshold', type=float, default=0.92,
                        help='Cosine similarity threshold for merging (default: 0.92)')
    parser.add_argument('--types', nargs='+',
                        default=['cause_factor', 'contributing_factor', 'weather_condition',
                                 'aircraft_system', 'phase_of_flight', 'maintenance_issue'],
                        help='Schema types to canonicalize')
    parser.add_argument('--dry-run', action='store_true')
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

    degree = build_degree_map(edges)

    # Collect entities by schema_type
    entities_by_type = collections.defaultdict(dict)  # type → {name: node_props}
    for e in edges:
        for side in ('start_node', 'end_node'):
            node = e[side]
            if node['label'] != 'entity':
                continue
            props = node['properties']
            st = props.get('schema_type', '')
            name = props.get('name', '')
            if st in args.types and name:
                entities_by_type[st][name] = props

    print(f"\nLoading embedding model ...")
    model = get_embedder()

    # Build rename_map: old_name → canonical_name
    rename_map = {}  # name → canonical_name

    for schema_type in args.types:
        ents = entities_by_type.get(schema_type, {})
        if len(ents) < 2:
            continue

        names = list(ents.keys())
        print(f"  Embedding {len(names):,} '{schema_type}' entities ...")
        t0 = time.time()
        embeddings = embed_batch(model, names)
        print(f"    {time.time()-t0:.1f}s")

        merge_groups = find_merge_groups(names, embeddings, args.threshold, schema_type)

        for rep_idx, member_idxs in merge_groups.items():
            member_names = [names[i] for i in member_idxs]
            # Canonical = highest degree (most connected node wins)
            canonical = max(member_names, key=lambda n: degree.get(n, 0))
            for name in member_names:
                if name != canonical:
                    rename_map[name] = canonical
                    if not args.dry_run:
                        pass  # will apply below

    print(f"\nTotal rename mappings: {len(rename_map)}")
    if args.dry_run or not rename_map:
        print("Sample renames:")
        for old, new in list(rename_map.items())[:20]:
            print(f"  '{old}' → '{new}'")
        if args.dry_run:
            return

    # Apply rename_map to graph
    print("Applying renames to graph ...")
    def rename_node(node):
        props = node.get('properties', {})
        name = props.get('name', '')
        if name in rename_map:
            new_props = dict(props)
            new_props['name'] = rename_map[name]
            new_props['_aliases'] = props.get('_aliases', []) + [name]
            return dict(node, properties=new_props)
        return node

    new_edges = []
    renamed_count = 0
    for e in edges:
        new_start = rename_node(e['start_node'])
        new_end = rename_node(e['end_node'])
        if new_start is not e['start_node'] or new_end is not e['end_node']:
            renamed_count += 1
        new_edges.append(dict(e, start_node=new_start, end_node=new_end))

    # Deduplicate edges (merging may create duplicates)
    seen = set()
    deduped = []
    for e in new_edges:
        key = (e['start_node']['properties'].get('name'),
               e['relation'],
               e['end_node']['properties'].get('name'))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    removed = len(new_edges) - len(deduped)
    print(f"  {renamed_count:,} edges renamed, {removed:,} duplicate edges removed")
    print(f"  Graph size: {len(edges):,} → {len(deduped):,} edges")

    print(f"Saving graph ...")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"✅ Canonicalized graph saved to {graph_path}")

    # Save rename map for inspection
    map_path = graph_path.replace('.json', '_rename_map.json')
    with open(map_path, 'w') as f:
        json.dump(rename_map, f, indent=2)
    print(f"   Rename map saved to {map_path}")


if __name__ == '__main__':
    main()
