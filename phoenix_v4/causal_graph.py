"""Causal Graph for Phoenix V4

Implements Pearl's causal framework using do-calculus for recommendation debiasing.

Key components:
- Causal DAG (Directed Acyclic Graph)
- Backdoor criterion for confounder identification
- Do-calculus for causal effect estimation

IMPORTANT: Protected attributes NOT in causal graph.
They are ONLY used for fairness testing, NEVER for predictions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CausalGraph:
    """Causal DAG for recommendation system.

    Defines causal relationships between variables in the recommendation system.

    Example DAG:
        user_activity -> propensity_score -> exposure -> engagement
        post_quality -> engagement
        position -> propensity_score, engagement (confounder!)
        popularity -> propensity_score, engagement (confounder!)
    """

    def __init__(self):
        """Initialize causal graph."""

        # Define causal structure (DAG)
        # Format: parent -> [children]
        self.dag: Dict[str, List[str]] = {
            # Behavioral factors -> Propensity
            'user_activity': ['propensity_score'],
            'post_quality': ['propensity_score', 'engagement'],
            'content_relevance': ['propensity_score', 'engagement'],

            # Confounders (affect both exposure and outcome)
            'position': ['propensity_score', 'engagement'],  # Confoun der!
            'popularity': ['propensity_score', 'engagement'],  # Confounder!
            'time_of_day': ['propensity_score', 'engagement'],  # Confounder!
            'recency': ['propensity_score', 'engagement'],  # Confounder!

            # Causal path
            'propensity_score': ['exposure'],
            'exposure': ['engagement'],

            # CRITICAL: Protected attributes NOT in causal graph
            # They are ONLY for testing, NEVER for predictions
        }

        # Reverse graph for finding parents
        self.reverse_dag: Dict[str, List[str]] = {}
        for parent, children in self.dag.items():
            for child in children:
                if child not in self.reverse_dag:
                    self.reverse_dag[child] = []
                self.reverse_dag[child].append(parent)

        logger.info("CausalGraph initialized (NO protected attributes in graph)")

    def get_parents(self, node: str) -> List[str]:
        """Get parents of a node.

        Args:
            node: Node name

        Returns:
            parents: List of parent nodes
        """
        return self.reverse_dag.get(node, [])

    def get_children(self, node: str) -> List[str]:
        """Get children of a node.

        Args:
            node: Node name

        Returns:
            children: List of child nodes
        """
        return self.dag.get(node, [])

    def find_paths(
        self,
        start: str,
        end: str,
        visited: Optional[Set[str]] = None
    ) -> List[List[str]]:
        """Find all paths from start to end.

        Args:
            start: Start node
            end: End node
            visited: Set of visited nodes (for cycle detection)

        Returns:
            paths: List of paths (each path is a list of nodes)
        """
        if visited is None:
            visited = set()

        if start == end:
            return [[start]]

        if start in visited:
            return []

        visited = visited | {start}
        paths = []

        for child in self.get_children(start):
            for path in self.find_paths(child, end, visited):
                paths.append([start] + path)

        return paths

    def identify_backdoor_paths(
        self,
        treatment: str,
        outcome: str
    ) -> List[List[str]]:
        """Find backdoor paths using backdoor criterion.

        Backdoor paths are non-causal paths from treatment to outcome
        that create spurious correlation.

        A backdoor path:
        1. Contains an arrow INTO the treatment node
        2. Does not contain any arrows OUT of the treatment node (except the first)

        Args:
            treatment: Treatment variable (e.g., 'exposure')
            outcome: Outcome variable (e.g., 'engagement')

        Returns:
            backdoor_paths: List of backdoor paths
        """
        backdoor_paths = []

        # Get parents of treatment (sources of backdoor paths)
        parents = self.get_parents(treatment)

        for parent in parents:
            # Find paths from parent to outcome
            paths = self.find_paths(parent, outcome)

            for path in paths:
                # Check if this is a backdoor path
                # It should not go through treatment (except at the start)
                if treatment not in path[1:]:
                    backdoor_paths.append([treatment] + path)

        return backdoor_paths

    def compute_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> Set[str]:
        """Compute minimal adjustment set using backdoor criterion.

        The adjustment set is the minimal set of variables to control for
        to block all backdoor paths.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            adjustment_set: Set of variables to condition on
        """
        backdoor_paths = self.identify_backdoor_paths(treatment, outcome)

        if len(backdoor_paths) == 0:
            return set()

        # Collect all confounders (nodes on backdoor paths)
        confounders = set()
        for path in backdoor_paths:
            # Confounders are nodes on the path (except treatment and outcome)
            for node in path:
                if node != treatment and node != outcome:
                    confounders.add(node)

        return confounders

    def do_operator(
        self,
        intervention: Dict[str, float],
        model: Callable,
        data: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute causal effect using do-calculus.

        P(Y | do(X=x)) - "What is Y if we SET X to x?"

        Different from P(Y | X=x) - "What is Y when we OBSERVE X=x?"

        The do-operator simulates an intervention by:
        1. Removing all incoming edges to X (breaking confounding)
        2. Setting X = x
        3. Computing P(Y | X=x) in the mutilated graph

        Args:
            intervention: Dict of {variable: value} to intervene on
            model: Prediction model function
            data: Observational data

        Returns:
            causal_effect: Predicted outcome under intervention
        """
        # Create intervened data
        intervened_data = data.copy()

        for variable, value in intervention.items():
            # Set variable to intervention value
            intervened_data[variable] = jnp.full_like(
                data[variable],
                value
            )

        # Run model on intervened data
        causal_effect = model(intervened_data)

        return causal_effect

    def frontdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        model: Callable,
        data: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Frontdoor criterion for unmeasured confounding.

        When there's an unmeasured confounder but we have a mediator M:
        X -> M -> Y, with unmeasured U -> X, U -> Y

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            mediator: Mediator variable
            model: Prediction model
            data: Observational data

        Returns:
            causal_effect: Estimated causal effect
        """
        # Step 1: Estimate P(M | do(X=x))
        # This is identified because no backdoor path from X to M
        do_x_to_m = self.do_operator(
            {treatment: data[treatment]},
            lambda d: d[mediator],
            data
        )

        # Step 2: Estimate P(Y | do(M=m))
        # This requires adjusting for X to block backdoor X <- U -> Y
        do_m_to_y = self.do_operator(
            {mediator: do_x_to_m},
            model,
            data
        )

        return do_m_to_y

    def validate_dag(self) -> bool:
        """Validate that graph is acyclic (no cycles).

        Returns:
            is_valid: True if DAG is valid (acyclic)
        """
        # Topological sort - if successful, graph is acyclic
        try:
            self._topological_sort()
            return True
        except ValueError:
            return False

    def _topological_sort(self) -> List[str]:
        """Perform topological sort (Kahn's algorithm).

        Returns:
            sorted_nodes: Topologically sorted nodes

        Raises:
            ValueError: If graph has cycles
        """
        # Compute in-degrees
        in_degree = {}
        all_nodes = set(self.dag.keys()) | set(self.reverse_dag.keys())

        for node in all_nodes:
            in_degree[node] = len(self.get_parents(node))

        # Queue of nodes with no incoming edges
        queue = [node for node in all_nodes if in_degree[node] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            # Reduce in-degree of children
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(all_nodes):
            raise ValueError("Graph has cycles!")

        return sorted_nodes

    def visualize(self) -> str:
        """Generate text visualization of DAG.

        Returns:
            viz: Text representation of DAG
        """
        lines = ["Causal DAG:"]
        lines.append("=" * 50)

        for parent, children in sorted(self.dag.items()):
            for child in children:
                lines.append(f"{parent} -> {child}")

        lines.append("=" * 50)
        lines.append(f"Total nodes: {len(set(self.dag.keys()) | set(self.reverse_dag.keys()))}")
        lines.append(f"Total edges: {sum(len(children) for children in self.dag.values())}")

        # Identify confounders
        confounders = self.compute_adjustment_set('exposure', 'engagement')
        lines.append(f"Confounders (exposure -> engagement): {confounders}")

        return "\n".join(lines)

    @classmethod
    def from_json(cls, json_path: str) -> "CausalGraph":
        """Load causal graph from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            graph: CausalGraph instance
        """
        import json

        with open(json_path, 'r') as f:
            dag_dict = json.load(f)

        graph = cls()
        graph.dag = dag_dict

        # Rebuild reverse DAG
        graph.reverse_dag = {}
        for parent, children in graph.dag.items():
            for child in children:
                if child not in graph.reverse_dag:
                    graph.reverse_dag[child] = []
                graph.reverse_dag[child].append(parent)

        return graph


def estimate_causal_effect(
    treatment_values: jnp.ndarray,
    outcome_values: jnp.ndarray,
    confounder_values: Dict[str, jnp.ndarray],
    method: str = "ipw"
) -> float:
    """Estimate causal effect using various methods.

    Args:
        treatment_values: Treatment variable values
        outcome_values: Outcome variable values
        confounder_values: Dict of confounder variable values
        method: Estimation method (ipw, regression, doubly_robust)

    Returns:
        causal_effect: Estimated average causal effect
    """
    if method == "ipw":
        # Inverse Propensity Weighting
        # Estimate propensity scores
        propensity_scores = estimate_propensity_scores(
            treatment_values,
            confounder_values
        )

        # IPW estimator
        weights = treatment_values / propensity_scores + \
                  (1 - treatment_values) / (1 - propensity_scores)

        causal_effect = jnp.mean(weights * outcome_values)

    elif method == "regression":
        # Regression adjustment
        # Fit outcome model: E[Y | T, C]
        # Then compute: E[Y | T=1, C] - E[Y | T=0, C]
        # Simplified version
        causal_effect = jnp.mean(outcome_values[treatment_values == 1]) - \
                       jnp.mean(outcome_values[treatment_values == 0])

    elif method == "doubly_robust":
        # Doubly robust estimator
        # Combines IPW and regression
        # More robust to model misspecification
        propensity_scores = estimate_propensity_scores(
            treatment_values,
            confounder_values
        )

        # Simplified version
        ate_ipw = jnp.mean(
            treatment_values * outcome_values / propensity_scores -
            (1 - treatment_values) * outcome_values / (1 - propensity_scores)
        )

        causal_effect = ate_ipw

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(causal_effect)


def estimate_propensity_scores(
    treatment: jnp.ndarray,
    confounders: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Estimate propensity scores P(T=1 | C).

    Args:
        treatment: Treatment values (0 or 1)
        confounders: Confounder values

    Returns:
        propensity_scores: Estimated P(T=1 | C)
    """
    # Simplified logistic regression (in production, use proper fitting)
    # For now, use empirical proportion
    propensity = jnp.mean(treatment)

    # Clip to avoid extreme values
    propensity = jnp.clip(propensity, 0.01, 0.99)

    propensity_scores = jnp.full_like(treatment, propensity, dtype=jnp.float32)

    return propensity_scores


def test_causal_graph():
    """Test causal graph functionality."""

    # Create causal graph
    graph = CausalGraph()

    print("Causal Graph Test")
    print("=" * 50)

    # Visualize
    print(graph.visualize())
    print()

    # Test backdoor paths
    print("Backdoor paths (exposure -> engagement):")
    backdoor_paths = graph.identify_backdoor_paths('exposure', 'engagement')
    for i, path in enumerate(backdoor_paths):
        print(f"  Path {i+1}: {' -> '.join(path)}")
    print()

    # Test adjustment set
    print("Adjustment set (confounders to control for):")
    adjustment_set = graph.compute_adjustment_set('exposure', 'engagement')
    print(f"  {adjustment_set}")
    print()

    # Validate DAG
    is_valid = graph.validate_dag()
    print(f"DAG is valid (acyclic): {is_valid}")
    print()

    assert is_valid, "DAG should be acyclic!"
    assert len(adjustment_set) > 0, "Should have confounders!"
    print("✓ Causal graph test passed!")


if __name__ == "__main__":
    test_causal_graph()
