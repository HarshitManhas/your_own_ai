"""Quick smoke-test of all API endpoints."""
import urllib.request
import json


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def delete(url):
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


Q = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06"
BASE = "http://localhost:8080"

passed = 0

# 1. /vectors
d = get(f"{BASE}/vectors")
assert d["count"] == 20
assert "pca" in d["vectors"][0]
print("PASS /vectors: 20 vectors with PCA coords")
passed += 1

# 2-4. /search for each algo
for algo in ["bruteforce", "kdtree", "hnsw"]:
    d = get(f"{BASE}/search?v={Q}&k=3&metric=cosine&algo={algo}")
    top = d["results"][0]["metadata"]
    print(f"PASS /search {algo}: top={top[:35]}")
    passed += 1

# 5. /benchmark
d = post(f"{BASE}/benchmark", {"v": Q, "k": 5, "metric": "cosine"})
algos = sorted(x["algo"] for x in d)
assert algos == ["bruteforce", "hnsw", "kdtree"], algos
timings = {x["algo"]: x["latencyUs"] for x in d}
print(f"PASS /benchmark: {timings}")
passed += 1

# 6. /insert + /delete
d = post(f"{BASE}/insert", {"metadata": "Test", "category": "test", "embedding": [0.1] * 16})
nid = d["id"]
d2 = delete(f"{BASE}/delete/{nid}")
assert d2["deleted"] == nid
print(f"PASS /insert + /delete: id={nid}")
passed += 1

# 7. /ollama/status
d = get(f"{BASE}/ollama/status")
assert "available" in d
print(f"PASS /ollama/status: available={d['available']}, model={d['embedModel']}")
passed += 1

# 8. homepage
with urllib.request.urlopen(f"{BASE}/") as r:
    html = r.read().decode()
assert "VectorDB" in html
print(f"PASS / homepage: {len(html)} bytes of HTML")
passed += 1

# 9. /doc/list
d = get(f"{BASE}/doc/list")
assert "documents" in d
print(f"PASS /doc/list: {d['count']} documents")
passed += 1

print(f"\n=== {passed}/9 TESTS PASSED ===")
