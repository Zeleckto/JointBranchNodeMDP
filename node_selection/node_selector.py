"""
node_selection/node_selector.py

BCNodeSelector — π2 node selector plugin for SCIP.

Gives HIGHEST priority to the child preferred by π2 (set by branch rule).
All other open nodes → SCIP hybridestim handles as normal.

If preferred_child was pruned before nodeselect fires, falls through to SCIP.
"""

from pyscipopt import Nodesel


class BCNodeSelector(Nodesel):
    """
    Single-shot child preference node selector.

    The branch rule (LearnedBranchRule) sets self.preferred_child after
    each branchVar() call. nodeselect() returns that child immediately
    on the next call, then clears preferred_child.

    Other open nodes are never scored by π2 — the action space of π2
    is always exactly {L, R} of the most recent branch.
    """

    def __init__(self):
        self.preferred_child = None   # int: SCIP node number, or None

    def nodeselect(self):
        """
        Called by SCIP to select the next node to process.
        Returns preferred child if available, else SCIP's best leaf.
        """
        if self.preferred_child is not None:
            children = self.model.getChildren()
            for child in children:
                if child.getNumber() == self.preferred_child:
                    self.preferred_child = None   # consume — one-shot
                    return {"selnode": child}

            # Preferred child not found — was pruned immediately after branch
            self.preferred_child = None
            # Fall through to SCIP default

        # All other cases: backtracking, post-pruning, etc.
        return {"selnode": self.model.getBestLeaf()}

    def nodecomp(self, node1, node2):
        """
        Pairwise comparison for SCIP's internal priority queue.
        Return 0 — let SCIP's default handle ordering of non-preferred nodes.
        """
        return 0
