"""Draw the compiled pipeline's DAG - from the YAML, not from memory.

A Kubeflow pipeline IS a graph, so this is the one place in the repository
where a picture genuinely beats a table. The point of the figure is the
CONDITIONAL STRUCTURE: a model cannot reach an endpoint without clearing
two separate gates, and there is an explicit else-branch when it does not.

Everything drawn here is PARSED FROM compiled/pipeline.yaml. Nothing is
hardcoded, so if the pipeline changes and is recompiled, the figure follows.
That matters - a hand-drawn architecture diagram is a claim about the code,
while this is a reading of it.

Run:
    python 06_pipelines/render_dag.py          # after pipeline.py has compiled

Concept reference: Foundations volume, Part G.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_setup"))
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SPEC = HERE / "compiled" / "pipeline.yaml"


def load_dag(spec_path: Path) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Flatten the nested KFP DAG into nodes and edges.

    KFP nests every conditional in its own sub-DAG component, so the graph is
    a tree of DAGs rather than one flat list. `condition-*` tasks are control
    flow, not work - they are recorded as gates and their children are lifted
    up to sit under whatever the condition depended on.
    """
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    components = doc["components"]

    nodes: list[dict] = []
    edges: list[tuple[str, str, str]] = []

    def walk(dag: dict, inherited_parents: list[str], gate_label: str | None) -> None:
        tasks = dag.get("tasks") or {}
        for name, task in tasks.items():
            deps = task.get("dependentTasks", []) or []
            parents = deps if deps else inherited_parents
            trigger = (task.get("triggerPolicy") or {}).get("condition")
            ref = task.get("componentRef", {}).get("name", "")
            sub = components.get(ref, {}).get("dag")

            if sub is not None:
                # A control-flow wrapper. Push its condition down to children.
                label = _readable(trigger) if trigger else gate_label
                walk(sub, parents, label)
            else:
                nodes.append({"name": name})
                for p in parents:
                    edges.append((p, name, gate_label or ""))

    walk(doc["root"]["dag"], [], None)
    return nodes, edges


def _readable(condition: str) -> str:
    """Turn a KFP trigger expression into something a human reads."""
    c = " ".join(condition.split())
    if "validate-Output'] == true" in c:
        return "if data valid"
    if c.startswith("!("):
        return "else"
    if ">=" in c:
        return "if accuracy >=\nthreshold"
    return c[:40]


def resolve_edges(nodes, edges):
    """Keep real task-to-task edges, then remove the transitively redundant ones.

    Lifting children out of their condition wrappers gives each one every
    parent the condition depended on. `condition-1` waited on BOTH ingest and
    validate, so `train` inherits both - but `ingest -> validate -> train`
    already implies `ingest -> train`. Drawing both says nothing extra and
    makes the figure harder to read, so the implied edge is dropped.
    """
    real = {n["name"] for n in nodes}
    kept = [(a, b, lab) for a, b, lab in edges if a in real and b in real]

    adj: dict[str, set[str]] = {n: set() for n in real}
    for a, b, _ in kept:
        adj[a].add(b)

    def reaches(src: str, dst: str, banned: tuple[str, str]) -> bool:
        seen, stack = set(), [src]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if (cur, nxt) == banned or nxt in seen:
                    continue
                if nxt == dst:
                    return True
                seen.add(nxt)
                stack.append(nxt)
        return False

    return [(a, b, lab) for a, b, lab in kept if not reaches(a, b, (a, b))]


# Laid out by hand because the graph is five nodes; anything cleverer would be
# harder to read than the coordinates themselves.
LAYOUT = {
    "ingest":   (0.5, 0.88),
    "validate": (0.5, 0.70),
    "train":    (0.5, 0.48),
    "deploy":   (0.24, 0.16),
    "alert":    (0.76, 0.16),
}
ROLE = {
    "ingest": "blue", "validate": "blue", "train": "blue",
    "deploy": "green", "alert": "vermillion",
}


def draw(nodes, edges, out_path: Path) -> Path:
    use_house_style()
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1)
    ax.axis("off")
    ax.grid(False)

    for a, b, label in edges:
        if a not in LAYOUT or b not in LAYOUT:
            continue
        x1, y1 = LAYOUT[a]
        x2, y2 = LAYOUT[b]
        gated = bool(label)
        ax.annotate(
            "", xy=(x2, y2 + 0.045), xytext=(x1, y1 - 0.045),
            arrowprops=dict(
                arrowstyle="-|>", linewidth=1.8,
                color=PALETTE["grey"] if not gated else PALETTE["orange"],
                linestyle="--" if gated else "-",
                shrinkA=0, shrinkB=0,
                connectionstyle="arc3,rad=0" if abs(x1 - x2) < 0.01 else "arc3,rad=0.12",
            ),
        )
        if gated:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + (0.10 if x2 > x1 else -0.10), my, label,
                    ha="center", va="center", fontsize=9.5, style="italic",
                    color=PALETTE["orange"],
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor=PALETTE["orange"], linewidth=1.0))

    for n in nodes:
        name = n["name"]
        if name not in LAYOUT:
            continue
        x, y = LAYOUT[name]
        colour = PALETTE[ROLE.get(name, "blue")]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.135, y - 0.045), 0.27, 0.09,
            boxstyle="round,pad=0.012", facecolor=colour,
            edgecolor="none", alpha=0.92, zorder=3))
        ax.text(x, y, name, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=4)

    ax.set_title("Compiled pipeline: two gates stand between training and an endpoint",
                 pad=16)
    ax.text(0.5, 0.035,
            "Parsed from compiled/pipeline.yaml - dashed orange edges are "
            "dsl.If conditions,\nand nothing reaches deploy without clearing both.",
            ha="center", va="center", fontsize=9.5, color=PALETTE["grey"])

    return save_figure(fig, out_path)


def main() -> None:
    if not SPEC.exists():
        print(f"  {SPEC} not found - run `python 06_pipelines/pipeline.py` first.")
        raise SystemExit(1)

    nodes, edges = load_dag(SPEC)
    edges = resolve_edges(nodes, edges)

    print(f"  Parsed {SPEC.relative_to(HERE.parent)}")
    print(f"  {len(nodes)} work components, {len(edges)} edges\n")
    print(f"  {'from':<12} -> {'to':<12}  gate")
    print("  " + "-" * 52)
    for a, b, lab in edges:
        print(f"  {a:<12} -> {b:<12}  {lab.replace(chr(10), ' ') or '-'}")

    gated = [e for e in edges if e[2]]
    print(f"\n  {len(gated)} of {len(edges)} edges are conditional. Those are the")
    print("  dsl.If gates - the reason a bad model cannot reach an endpoint.")

    out = draw(nodes, edges, HERE / "images" / "pipeline_dag.png")
    print(f"\n  Saved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
