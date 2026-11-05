"""Frozen E2 actual gate support: connected components versus cliques."""
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/sp_20261008_120638_983201/e2_pgct_s1/communication.jsonl'
sizes, patterns, cliques, samples = Counter(), Counter(), Counter(), {}
updates = asymmetric = 0
for line in SOURCE.open(encoding='utf8'):
    row = json.loads(line)
    w = row['transfer_weights']
    adj = [[i != j and (w[i][j] > 0 or w[j][i] > 0) for j in range(4)] for i in range(4)]
    asymmetric += any((w[i][j] > 0) != (w[j][i] > 0) for i, j in combinations(range(4), 2))
    remaining, groups = set(range(4)), []
    while remaining:
        seen = {next(iter(remaining))}
        todo = list(seen)
        while todo:
            i = todo.pop()
            for j in range(4):
                if adj[i][j] and j not in seen:
                    seen.add(j)
                    todo.append(j)
        remaining -= seen
        groups.append(sorted(seen))
    assert sum(map(len, groups)) == 4
    sizes[max(map(len, groups))] += 1
    patterns[str(tuple(sorted(map(len, groups))))] += 1
    updates += 1
    for group in groups:
        if len(group) < 3:
            continue
        full = all(adj[i][j] for i, j in combinations(group, 2))
        key = f'size_{len(group)}_clique_{full}'
        cliques[key] += 1
        if key not in samples:
            samples[key] = dict(env_steps=row['env_steps'], members=[i+1 for i in group],
                edges=[[i+1, j+1] for i, j in combinations(range(4), 2) if adj[i][j]],
                ema={f'{i+1}-{j+1}': row['policy_distance_ema'][i][j] for i, j in combinations(range(4), 2)})
result = dict(source=str(SOURCE), updates=updates, denominator='All logged updates, including warmup and no-edge updates',
    graph='Undirected support of actual transfer_weights, either direction > 0',
    largest_component_counts=dict(sizes), component_size_patterns=dict(patterns),
    component_clique_counts=dict(cliques), asymmetric_support_updates=asymmetric, examples=samples,
    limitation='Graph connectivity does not imply task-team identity, all-pair similarity, or beneficial critic transfer.')
assert updates == 3907
assert sum(sizes.values()) == updates
assert cliques['size_3_clique_True'] + cliques['size_3_clique_False'] == sizes[3]
assert cliques['size_4_clique_False'] == sizes[4]
assert cliques.get('size_4_clique_True', 0) == 0
Path(__file__).with_name('e2_components.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
