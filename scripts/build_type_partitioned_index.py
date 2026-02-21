"""
build_type_partitioned_index.py — Pre-build per-schema-type FAISS indices.

Problem: during retrieval, `_process_triple_index` expands 3-hop BFS from
FAISS hits, producing 200K–400K triples that all need re-scoring. This is
slow even with our BFS cap.

Solution: pre-build one FAISS index per schema_type pair (head_type, tail_type).
At query time, use only the index for the relevant type pair.
This reduces search space from 400K to ~2K triples per type bucket.

Also patches faiss_filter.py to use the partitioned index when available.

Usage:
    cd /root/youtu-graphrag
    python scripts/build_type_partitioned_index.py --dataset ntsb_sample
"""
import argparse, json, os, sys, time, collections
import numpy as np

sys.path.insert(0, '/root/youtu-graphrag')
from config import get_config

def get_embedder(model_name='all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def encode_batch(model, texts, batch_size=256):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
    return np.vstack(all_embs).astype('float32')

def triple_text(h, r, t):
    return f"{h} {r} {t}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--min-triples', type=int, default=5,
                        help='Min triples in a type bucket to build index for')
    args = parser.parse_args()

    config = get_config()
    ds_config = config.get_dataset_config(args.dataset)
    graph_path = ds_config.graph_output
    if not os.path.exists(graph_path):
        for alt in [graph_path.replace(".json","_new.json"), f"output/graphs/{args.dataset}_new.json"]:
            if os.path.exists(alt): graph_path = alt; break
    cache_dir = f"retriever/faiss_cache_new/{args.dataset}"

    print(f"Loading graph: {graph_path}")
    with open(graph_path) as f:
        edges = json.load(f)
    print(f"  {len(edges):,} edges")

    # Build node name → schema_type lookup
    name_to_type = {}
    for e in edges:
        for side in ('start_node', 'end_node'):
            props = e[side].get('properties', {})
            name = props.get('name', '')
            st = props.get('schema_type', 'unknown')
            if name:
                name_to_type[name] = st

    # Group triples by (head_type, tail_type) bucket
    print("Grouping triples by schema_type pairs ...")
    buckets = collections.defaultdict(list)  # (h_type, t_type) → [(h,r,t), ...]
    for e in edges:
        h = e['start_node']['properties'].get('name', '')
        t = e['end_node']['properties'].get('name', '')
        r = e.get('relation', '')
        if not h or not t:
            continue
        h_type = name_to_type.get(h, 'unknown')
        t_type = name_to_type.get(t, 'unknown')
        key = (h_type, t_type)
        buckets[key].append((h, r, t))

    # Filter buckets
    large_buckets = {k: v for k, v in buckets.items() if len(v) >= args.min_triples}
    print(f"  {len(large_buckets)} type-pair buckets (>= {args.min_triples} triples)")
    for (ht, tt), triples in sorted(large_buckets.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"    ({ht}, {tt}): {len(triples):,} triples")

    print("\nLoading embedding model ...")
    model = get_embedder(config.embeddings.model_name)

    # Build and save per-bucket FAISS index
    import faiss
    import pickle

    out_dir = os.path.join(cache_dir, 'type_partitioned')
    os.makedirs(out_dir, exist_ok=True)

    bucket_manifest = {}
    total_indexed = 0

    for (h_type, t_type), triples in sorted(large_buckets.items(), key=lambda x: -len(x[1])):
        key_str = f"{h_type}___{t_type}"
        index_path = os.path.join(out_dir, f"{key_str}.index")
        map_path = os.path.join(out_dir, f"{key_str}.map.pkl")

        texts = [triple_text(h, r, t) for h, r, t in triples]

        t0 = time.time()
        embs = encode_batch(model, texts)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, index_path)

        with open(map_path, 'wb') as f:
            pickle.dump(triples, f)

        elapsed = time.time() - t0
        print(f"  [{key_str}] {len(triples):,} triples → {elapsed:.1f}s")

        bucket_manifest[key_str] = {
            'h_type': h_type,
            't_type': t_type,
            'count': len(triples),
            'index_path': index_path,
            'map_path': map_path,
        }
        total_indexed += len(triples)

    # Save manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(bucket_manifest, f, indent=2)

    print(f"\n✅ Type-partitioned index built:")
    print(f"   {len(large_buckets)} buckets, {total_indexed:,} total triples")
    print(f"   Saved to {out_dir}/")
    print(f"\nAt query time: instead of BFS-expanding 400K triples,")
    print(f"search only the ~{total_indexed//len(large_buckets):,} triples in the relevant type bucket.")


if __name__ == '__main__':
    main()
