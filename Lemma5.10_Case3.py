"""Exhaustive SAT verification for Case 3 of Lemma 5.10.

We search for group-ring elements

    D = 1 - zeta_3 (v + ... + v^4) - zeta_3^2 (u + ... + u^4)
        + sum_{1 <= i,j <= 4} a[i,j] u^i v^j,

where every a[i,j] is a fourth root of unity and a[1,1] = 1, which corresponds to 
BH(C_5 x C_5, 12) matrices.  
The program imposes the group-invariant condition

    sum_{g in G} a_{g+h} * conjugate(a_g) = 0    for every nonidentity h in G = C_5 x C_5,

as well as the four stated nonconstant-entries conditions. It then checks that the
six resulting solutions are exactly the character-product construction, up to scaling 
and equivalence, given in Theorem 5.1.

Requires: python-sat (PySAT), including its PB encoder.
"""

from itertools import combinations
from time import perf_counter
from typing import Dict, Iterable, Tuple

from pysat.pb import PBEnc
from pysat.solvers import Glucose4


Point = Tuple[int, int]
Matrix = Dict[Point, int]  # Entry exponents: k represents zeta_12^k.

# Coordinates of zeta_12^k in the Q-basis
# (1, zeta_12, zeta_12^2, zeta_12^3) of Q(zeta_12).
BASIS_COORDINATES = {
    0: (1, 0, 0, 0), 1: (0, 1, 0, 0), 2: (0, 0, 1, 0), 3: (0, 0, 0, 1),
    4: (-1, 0, 1, 0), 5: (0, -1, 0, 1), 6: (-1, 0, 0, 0),
    7: (0, -1, 0, 0), 8: (0, 0, -1, 0), 9: (0, 0, 0, -1),
    10: (1, 0, -1, 0), 11: (0, 1, 0, -1),
}

GROUP = [(i, j) for i in range(5) for j in range(5)]
FOURTH_ROOTS = (0, 3, 6, 9)  # Exponents for 1, i, -1, -i.

# Each set (corresponding to 4 different subgroups of C_5 x C_5) must not consist of four equal interior coefficients.
# These conditions filter out the finite translation plane construction by Schmidt, Wong, and Xiang (2021).
NONCONSTANT_SETS = [
    [(1, 1), (2, 2), (3, 3), (4, 4)],
    [(1, 2), (2, 4), (3, 1), (4, 3)],
    [(1, 3), (2, 1), (3, 4), (4, 2)],
    [(1, 4), (2, 3), (3, 2), (4, 1)],
]


def group_add(g: Point, h: Point) -> Point:
    """Addition in C_5 x C_5."""
    return ((g[0] + h[0]) % 5, (g[1] + h[1]) % 5)


def coefficient_domains() -> Dict[Point, Tuple[int, ...]]:
    """Return the allowed zeta_12 exponents for every coefficient a[i,j]."""
    domains = {}
    for i, j in GROUP:
        if (i, j) in {(0, 0), (1, 1)}:
            domains[i, j] = (0,)       # a[0,0] = 1 and the normalization a[1,1] = 1.
        elif j == 0:
            domains[i, j] = (10,)      # -zeta_3 = zeta_12^10.
        elif i == 0:
            domains[i, j] = (2,)       # -zeta_3^2 = zeta_12^2.
        else:
            domains[i, j] = FOURTH_ROOTS
    return domains


def add_exactly_one(solver: Glucose4, literals: Iterable[int]) -> None:
    """Require exactly one literal, using a direct pairwise encoding. 
    This is used later to encode exactly one value for each entry."""
    literals = list(literals)
    solver.add_clause(literals)                     # Encode at least one literal.
    for left, right in combinations(literals, 2):   # Encode at most one literal.
        solver.add_clause([-left, -right])


def add_nonconstant_conditions(solver, domains, root_literal) -> None:
    """Add the four required conditions that the listed entries are not constant."""
    for cells in NONCONSTANT_SETS:
        for root in FOURTH_ROOTS:
            # If a root is not allowed at some cell, equality to that root is
            # already impossible and no clause is needed.
            if all(root in domains[cell] for cell in cells):
                solver.add_clause([-root_literal[cell, root] for cell in cells])


def expected_character_matrices() -> list[Matrix]:
    """Return the six character-product matrices filtered by the four conditions."""
    # Discrete logarithms in Z_5^x for the generator 2:
    # log_2(1), log_2(2), log_2(3), log_2(4) = 0, 1, 3, 2.
    discrete_log = (0, 4, 1, 3, 2)
    characters = (3, 6, 9)  # The three nontrivial characters of Z_5^x.

    candidates = [
        {(i, j): (r * discrete_log[i] + s * discrete_log[j]) % 12
         for i in range(1, 5) for j in range(1, 5)}
        for r in characters for s in characters
    ]
    return [
        matrix for matrix in candidates
        if all(len({matrix[cell] for cell in cells}) > 1 for cells in NONCONSTANT_SETS)
    ]


def build_solver():
    """Build the SAT/PB encoding and return its primary root-choice variables."""
    solver = Glucose4()
    domains = coefficient_domains()
    root_literal = {}
    next_variable = 0

    def fresh_variable() -> int:
        nonlocal next_variable
        next_variable += 1
        return next_variable

    # A[g,k] means a_g = zeta_12^k.  Variables are created only for allowed roots.
    for g, roots in domains.items():
        root_literals = []
        for root in roots:
            variable = fresh_variable()
            root_literal[g, root] = variable
            root_literals.append(variable)
        add_exactly_one(solver, root_literals)

    add_nonconstant_conditions(solver, domains, root_literal)

    # For each h \ne 1_G, encode \sum_{g \in G} a_{g+h} conjugate(a_g) = 0.
    for h in GROUP[1:]:
        # One pseudo-Boolean equation for each coordinate (0, 1, 2, 3) 
        # in the chosen basis (1, zeta_12, zeta_12^2, zeta_12^3).
        equations = [([], []) for _ in range(4)]  # (literals, integer weights)

        for g in GROUP:
            # u = g + h
            u = group_add(g, h)
            roots_at_u, roots_at_g = domains[u], domains[g]
            reachable_differences = {(p - q) % 12 for p in roots_at_u for q in roots_at_g}

            # difference[k] means a_u conjugate(a_g) = zeta_12^k.
            difference = {k: fresh_variable() for k in reachable_differences}

            # Selected roots imply their quotient's difference variable.
            # (a_u = zeta_12^p AND a_g = zeta^q) => a_u conjugate(a_g) = zeta_12^{p - q}
            for p in roots_at_u:
                for q in roots_at_g:
                    solver.add_clause([
                        -root_literal[u, p],
                        -root_literal[g, q],
                        difference[(p - q) % 12],
                    ])

            # Conversely, a difference variable fixes a_g once the
            # a_u is selected. This makes each difference deterministic.
            # (a_u = zeta_12^p AND a_u conjugate(a_g) = zeta_12^{p - q}) => a_g = zeta^q
            for k, difference_literal in difference.items():
                for p in roots_at_u:
                    q = (p - k) % 12
                    if q in roots_at_g:
                        solver.add_clause([
                            -difference_literal,
                            -root_literal[u, p],
                            root_literal[g, q],
                        ])
                    else:
                        solver.add_clause([-difference_literal, -root_literal[u, p]])

            # If difference[k] is true, then a_u * conjugate(a_g) = zeta_12^k. Expand zeta_12^k 
            # in the basis (1, zeta_12, zeta_12^2, zeta_12^3), and add its nonzero coordinates 
            # to the four equations for \sum_{g \in G} a_{g+h} * conjugate(a_g) = 0.
            for k, difference_literal in difference.items():
                for coordinate, weight in enumerate(BASIS_COORDINATES[k]):
                    if weight:
                        equations[coordinate][0].append(difference_literal)
                        equations[coordinate][1].append(weight)

        # Convert all four weighted equalities to CNF, allocating PB auxiliaries
        # strictly after the variables already allocated above.
        for literals, weights in equations:
            # encodes Σᵢ weights[i] · literals[i] = 0
            cnf = PBEnc.equals(lits=literals, weights=weights, bound=0, top_id=next_variable) 
            next_variable = cnf.nv
            solver.append_formula(cnf.clauses)

    return solver, domains, root_literal, next_variable


def extract_interior_matrix(model, domains, root_literal) -> Matrix:
    """Extract the 4 x 4 interior exponent matrix (a_g entries where g[0], g[1] are not 0) 
    from one SAT model."""
    positive_literals = set(model)
    return {
        g: next(root for root in domains[g] if root_literal[g, root] in positive_literals)
        for g in GROUP if g[0] != 0 and g[1] != 0
    }


def main() -> None:
    started = perf_counter()
    solver, domains, root_literal, variable_count = build_solver()
    expected = expected_character_matrices()

    print(f"Formula built with {variable_count} variables.")
    print(f"Searching for the {len(expected)} expected product-character solutions...")

    solutions = 0
    while solver.solve():
        model = solver.get_model()
        matrix = extract_interior_matrix(model, domains, root_literal)
        solutions += 1

        if matrix in expected:
            print(f"Solution {solutions}: product-character solution.")
        else:
            print(f"Solution {solutions}: COUNTEREXAMPLE\n{matrix}")

        # Block this coefficient assignment only. PB encodings have auxiliary
        # variables, which must not create duplicate mathematical solutions.
        positive_literals = set(model)
        solver.add_clause([
            -literal if literal in positive_literals else literal
            for literal in root_literal.values()
        ])

    elapsed = perf_counter() - started
    print(f"\nSearch complete in {elapsed:.2f} seconds.")
    print(f"Total solutions: {solutions}")
    print("ALL SOLUTIONS ARE FROM THE PRODUCT-CHARACTER CONSTRUCTION. \nLEMMA 5.10 IS TRUE." if solutions == len(expected) 
          else "LEMMA 5.10 IS FALSE.")
    solver.delete()


if __name__ == "__main__":
    main()
