"""
demo_data.py — 20 pre-built 16-dimensional semantic vectors.

Hand-crafted embeddings across 4 categories:
    Dimensions  0– 3: Computer Science / Algorithms
    Dimensions  4– 7: Mathematics
    Dimensions  8–11: Food
    Dimensions 12–15: Sports

These are used for the demo visualization and benchmarking panel.
Equivalent to DemoData.java in the original Java project.
"""

from .vector_db import VectorDB


def load(db: VectorDB) -> None:
    """Insert 20 pre-built 16D demo vectors into the database."""

    # ── Computer Science / Algorithms ─────────────────────────────────────────
    db.insert("Linked List: nodes connected by pointers", "cs",
        [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06])
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10])
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07])
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07])
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09])

    # ── Mathematics ───────────────────────────────────────────────────────────
    db.insert("Calculus: derivatives integrals limits continuity", "math",
        [0.15, 0.12, 0.10, 0.18, 0.88, 0.85, 0.80, 0.76, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.07, 0.11])
    db.insert("Linear Algebra: matrices eigenvalues vector spaces", "math",
        [0.18, 0.14, 0.12, 0.16, 0.85, 0.90, 0.82, 0.78, 0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08])
    db.insert("Number Theory: primes modular arithmetic", "math",
        [0.12, 0.10, 0.14, 0.09, 0.82, 0.78, 0.88, 0.84, 0.06, 0.07, 0.08, 0.09, 0.07, 0.11, 0.09, 0.06])
    db.insert("Statistics: mean variance standard deviation", "math",
        [0.14, 0.16, 0.11, 0.13, 0.80, 0.84, 0.78, 0.90, 0.09, 0.08, 0.07, 0.06, 0.08, 0.09, 0.11, 0.07])
    db.insert("Probability: Bayes theorem distributions random variables", "math",
        [0.11, 0.13, 0.16, 0.12, 0.84, 0.80, 0.86, 0.82, 0.07, 0.10, 0.08, 0.07, 0.11, 0.08, 0.06, 0.09])

    # ── Food ──────────────────────────────────────────────────────────────────
    db.insert("Pizza: dough tomato sauce mozzarella basil oven", "food",
        [0.06, 0.09, 0.07, 0.08, 0.10, 0.07, 0.09, 0.06, 0.88, 0.85, 0.80, 0.78, 0.12, 0.10, 0.08, 0.14])
    db.insert("Sushi: rice fish nori vinegar wasabi", "food",
        [0.08, 0.06, 0.09, 0.07, 0.08, 0.09, 0.06, 0.10, 0.85, 0.90, 0.82, 0.76, 0.09, 0.07, 0.11, 0.08])
    db.insert("Pasta: durum wheat semolina gluten eggs al dente", "food",
        [0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.08, 0.07, 0.82, 0.78, 0.88, 0.84, 0.08, 0.12, 0.09, 0.06])
    db.insert("Curry: spices turmeric cumin coriander garam masala", "food",
        [0.09, 0.07, 0.08, 0.06, 0.07, 0.08, 0.10, 0.09, 0.80, 0.82, 0.84, 0.90, 0.10, 0.08, 0.07, 0.09])
    db.insert("Tacos: tortilla beans salsa guacamole cilantro", "food",
        [0.06, 0.10, 0.09, 0.07, 0.11, 0.07, 0.08, 0.08, 0.86, 0.80, 0.78, 0.84, 0.07, 0.09, 0.10, 0.08])

    # ── Sports ────────────────────────────────────────────────────────────────
    db.insert("Football: quarterback touchdown field goal offensive line", "sports",
        [0.08, 0.07, 0.09, 0.10, 0.09, 0.06, 0.07, 0.08, 0.11, 0.09, 0.07, 0.10, 0.88, 0.85, 0.80, 0.78])
    db.insert("Basketball: dribble three pointer slam dunk pick and roll", "sports",
        [0.10, 0.08, 0.07, 0.09, 0.06, 0.10, 0.09, 0.07, 0.09, 0.07, 0.10, 0.08, 0.85, 0.90, 0.82, 0.76])
    db.insert("Tennis: forehand backhand volley ace tiebreak", "sports",
        [0.07, 0.09, 0.10, 0.08, 0.08, 0.09, 0.06, 0.10, 0.07, 0.10, 0.09, 0.07, 0.82, 0.78, 0.88, 0.84])
    db.insert("Swimming: freestyle backstroke butterfly breaststroke", "sports",
        [0.09, 0.06, 0.08, 0.07, 0.07, 0.08, 0.10, 0.09, 0.08, 0.08, 0.06, 0.09, 0.80, 0.82, 0.84, 0.90])
    db.insert("Cricket: batsman bowler wicket innings over LBW", "sports",
        [0.06, 0.10, 0.07, 0.08, 0.10, 0.07, 0.08, 0.06, 0.10, 0.09, 0.08, 0.07, 0.86, 0.80, 0.78, 0.84])
