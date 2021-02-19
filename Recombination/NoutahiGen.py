#https://mrnoutahi.com/2017/12/05/How-to-simulate-a-tree/
#https://mrnoutahi.com/2017/12/05/How-to-simulate-a-tree/
#https://mrnoutahi.com/2017/12/05/How-to-simulate-a-tree/

#!/usr/bin/env python
from ete3 import Tree
import random


def delete_single_child_internal(t):
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        if(not node.is_leaf() and len(node.get_children()) < 2):
            node.delete()

    if len(t.get_children()) == 1:
        t = t.children[0]
        t.up = None


def birth_only_tree(birth, nsize=10, max_time=None):
    """Generates a uniform-rate birth-only tree.
    Arguments:
        - ``birth`` : birth rate
        - ``nsize``  :  desired number of leaves
        - ``max_time`` : maximum allowed time for evolution
    """

    done = False
    total_time = 0

    # create initial root node and set length to 0
    tree = Tree()
    tree.dist = 0.0
    # check if a stopping condition is provided
    if not (nsize or max_time):
        raise ValueError('A stopping criterion is required')

    while True:

        # get the current list of extant species
        leaf_nodes = tree.get_leaves()

        # get required waiting time before next speciation
        wtime = random.expovariate(len(leaf_nodes) / birth)

        # check if stopping criterion is reached then
        # stop loop if yes
        if len(leaf_nodes) >= nsize or (max_time and total_time + wtime >= max_time):
            done = True

        # update the length of all current leaves
        # while not exceeding the maximum evolution time
        max_limited_time = min(
            wtime, (max_time or total_time + wtime) - total_time)
        for leaf in leaf_nodes:
            leaf.dist += max_limited_time

        if done:
            break

        # update total time of evolution
        total_time += max_limited_time

        # add a new node to a randomly chosen leaf.
        node = random.choice(leaf_nodes)
        c1 = Tree()
        c2 = Tree()
        node.add_child(c1)
        node.add_child(c2)
        c1.dist = 0.0
        c2.dist = 0.0

    # Label leaves here and also update
    # the last branch lengths

    leaf_nodes = tree.get_leaves()
    leaf_compteur = 1
    for (ind, node) in enumerate(leaf_nodes):
        node.name = 'T%d' % leaf_compteur
        leaf_compteur += 1

    return tree


def birth_death_tree(birth, death, nsize=10, max_time=None, remlosses=True, r=True):
    """Generates a birth-death tree.
    Arguments:
        - ``birth`` : birth rate
        - ``death`` : death rate
        - ``nsize`` : desired number of leaves
        - ``max_time`` : maximum time of evolution
        - ``remlosses`` : whether lost leaves (extinct taxa) should be pruned from tree
        - ``r`` : repeat until success
    """
    # initialize tree with root node
    tree = Tree()
    tree.add_features(extinct=False)
    tree.dist = 0.0
    done = False

    # get current list of leaves
    leaf_nodes = tree.get_leaves()
    curr_num_leaves = len(leaf_nodes)

    total_time = 0
    died = set([])

    # total event rate to compute waiting time
    event_rate = float(birth + death)

    while True:
        # waiting time based on event_rate
        wtime = random.expovariate(event_rate)
        total_time += wtime
        for leaf in leaf_nodes:
            # extinct leaves cannot update their branches length
            if not leaf.extinct:
                leaf.dist += wtime

        if curr_num_leaves >= nsize:
            done = True

        if done:
            break

        # if event occurs within time constraints
        if max_time is None or total_time <= max_time:

            # select node at random, then find chance it died or give birth
            # (speciation)
            node = random.choice(leaf_nodes)
            eprob = random.random()
            leaf_nodes.remove(node)
            curr_num_leaves -= 1

            # birth event (speciation)
            if eprob < birth / event_rate:
                child1 = Tree()
                child1.dist = 0
                child1.add_features(extinct=False)
                child2 = Tree()
                child2.dist = 0
                child2.add_features(extinct=False)
                node.add_child(child1)
                node.add_child(child2)
                leaf_nodes.append(child1)
                leaf_nodes.append(child2)
                # update add two new leave
                # (remember that parent was removed)
                curr_num_leaves += 2

            else:
                # death of the chosen node
                if curr_num_leaves > 0:
                    node.extinct = True
                    died.add(node)
                else:
                    if not r:
                        raise ValueError(
                            "All lineage went extinct, please retry")
                    # Restart the simulation because the tree has gone
                    # extinct
                    tree = Tree()
                    leaf_nodes = tree.get_leaves()
                    tree.add_features(extinct=False)
                    tree.dist = 0.0
                    curr_num_leaves = 1
                    died = set([])
                    total_time = 0

            # this should always hold true
            assert curr_num_leaves == len(leaf_nodes)

    if remlosses:
        # prune lost leaves from tree
        leaves = set(tree.get_leaves()) - died
        tree.prune(leaves)
        # remove all non binary nodes
        delete_single_child_internal(tree)

    leaf_nodes = tree.get_leaves()
    leaf_compteur = 1
    for ind, node in enumerate(leaf_nodes):
        # label only extant leaves
        if not node.extinct:
            # node.dist += wtime
            node.name = "T%d" % leaf_compteur
            leaf_compteur += 1
    return tree


if __name__ == '__main__':
    yuletree = birth_only_tree(1.0, nsize=4)
    bdtree = birth_death_tree(1.0, 0.5, nsize=4)
    tree = yuletree
    print("tree",tree)
    for leaf in tree.get_leaves():
        print(leaf,leaf.dist)