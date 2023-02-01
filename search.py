# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graphSearchByChels(problem: SearchProblem, structure):
    """
    IMPLEMENT MY (chelsea's) SEARCH METHOD HERE!
    The "structure" var is used to represent the appropriate structure for
    the search method used (Stack, Queue, etc)
    """

    #visited: track which nodes have been visited (is there a better structure?)
    #start: initial state (first node)
    visited = []
    state = problem.getStartState()
    #create the initial node from the starting state (no action taken, no cost)
    structure.push([state, None, 0])

    #loop until all states have been addressed (i.e. since there's nothing to add, structure is empty)
    while not structure.isEmpty():
        #pop the next node for search
        node = structure.pop()
        #check if each node contains a goal state. if not, move onward
        for nextNode in problem.getSuccessors(node[0]):
            if problem.isGoalState(nextNode[0]):
                return nextNode #return goal node
            #if this is a new state, mark as visited and 
            if nextNode[0] not in visited:
                visited.add(nextNode[0])
                structure.push(nextNode)




def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    fringe.push((problem.getStartState(), []))
    closed = set()
    while not fringe.isEmpty():
        node, actions = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in closed:
            closed.add(node)
            for childNode, nextAction, _ in problem.getSuccessors(node):
                fringe.push((childNode, actions + [nextAction]))

    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))
    closed = set()
    while not fringe.isEmpty():
        node, actions = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in closed:
            closed.add(node)
            for childNode, nextAction, _ in problem.getSuccessors(node):
                fringe.push((childNode, actions + [nextAction]))

    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0), 0)
    closed = set()
    while not fringe.isEmpty():
        node, actions, totalCost = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in closed:
            closed.add(node)
            for childNode, nextAction, cost in problem.getSuccessors(node):
                fringe.push((childNode, actions + [nextAction], totalCost + cost), totalCost + cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    h = heuristic(start, problem)
    fringe.push((start, [], 0), h)
    closed = dict()

    while not fringe.isEmpty():
        node, actions, cost = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in closed or cost < closed[node]:
            closed[node] = cost
            for childNode, nextAction, gChild in problem.getSuccessors(node):
                fringe.push((childNode, actions + [nextAction], cost + gChild), 
                            cost + gChild + heuristic(childNode, problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
