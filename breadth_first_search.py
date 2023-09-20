
"""Basic algorithms for breadth-first searching the nodes of a graph."""
import networkx as nx
import copy
from collections import deque

__all__ = ['bfs_edges', 'bfs_tree', 'bfs_predecessors', 'bfs_successors']


def generic_bfs_edges(G, label, source, neighbors=None, depth_limit=None):
    """Iterate over edges in a breadth-first search.

    The breadth-first search begins at `source` and enqueues the
    neighbors of newly visited nodes specified by the `neighbors`
    function.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    neighbors : function
        A function that takes a newly visited node of the graph as input
        and returns an *iterator* (not just a list) of nodes that are
        neighbors of that node. If not specified, this is just the
        ``G.neighbors`` method, but in general it can be any function
        that returns an iterator over some or all of the neighbors of a
        given node, in any order.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth
    """
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    neigh = list(neighbors(source))
    #print(neigh)
    neighlabel = []
    for nei in neigh:
        neighlabel.append(label[int(nei)])
    #print(neighlabel)
    neighindex = sorted(range(len(neighlabel)), key=lambda k: neighlabel[k], reverse=True)
    sortedneighbor = []
    for ele in neighindex:
        sortedneighbor.append(neigh[ele])

    queue = deque([(source, depth_limit, iter(sortedneighbor))])
    #print(type(neighbors(source)))
    #print(list(iter(sortedneighbor)))

    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    chil = list(neighbors(child))
                    chillabel = []
                    for ch in chil:
                        chillabel.append(label[int(ch)])

                    chilindex = sorted(range(len(chillabel)), key=lambda k: chillabel[k], reverse=True)
                    sortedneighbor = []
                    for ele in chilindex:
                        sortedneighbor.append(chil[ele])
                    
                    queue.append((child, depth_now - 1, iter(sortedneighbor)))
        except StopIteration:
            queue.popleft()


def bfs_edges(G, label, source, reverse=False, depth_limit=None):
    """Iterate over edges in a breadth-first-search starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    edges: generator
       A generator of edges in the breadth-first-search.


    """
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    # TODO In Python 3.3+, this should be `yield from ...`
    for e in generic_bfs_edges(G, label, source, successors, depth_limit):
        yield e



def bfs_tree(G, source, reverse=False, depth_limit=None):
    """Return an oriented tree constructed from of a breadth-first-search
    starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    T: NetworkX DiGraph
       An oriented tree


    """
    T = nx.DiGraph()
    T.add_node(source)
    edges_gen = bfs_edges(G, source, reverse=reverse, depth_limit=depth_limit)
    T.add_edges_from(edges_gen)
    return T



def bfs_predecessors(G, source, depth_limit=None):
    """Returns an iterator of predecessors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    pred: iterator
        (node, predecessors) iterator where predecessors is the list of
        predecessors of the node.


    """
    for s, t in bfs_edges(G, source, depth_limit=depth_limit):
        yield (t, s)



def bfs_successors(G, source, depth_limit=None):
    """Returns an iterator of successors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search and return edges in
       the component reachable from source.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Returns
    -------
    succ: iterator
       (node, successors) iterator where successors is the list of
       successors of the node.

    """
    parent = source
    children = []
    for p, c in bfs_edges(G, source, depth_limit=depth_limit):
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)
