# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from searchAgents import mazeDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions()
        
        # identify max score using recursion to search tree
        max = 0
        index = 0
        counter = 0
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.getExpectedScore(successor, 1, self.depth)
            if score > max:
                max = score
                index = counter
            counter += 1
        
        return legalActions[index]

    #method to search tree for expected score
    def getExpectedScore(self, gameState, agentIndex: int, depth: int):
        
        #no search needed if nothing left to search
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        #identify legal actions and the next agent to make a move (rollover to 0
        # if all ghosts have taken their turn)
        legalActions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        if nextAgent > gameState.getNumAgents() - 1:
            nextAgent = 0
        
        #if rolling over to 0, we must be at the next depth of game states
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        pacmanMax = 0
        ghostSum = 0

        #for every possible action, generate a successor, find the score by
        #recursively calling this method
        #if pacman, take the maximum score from successors
        #if ghost, sum up scores to return an average (since equal probability)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.getExpectedScore(successor, nextAgent, nextDepth)

            if agentIndex == 0 and score > pacmanMax:
                pacmanMax = score
            else:
                ghostSum += score
            
        if agentIndex == 0:
            return pacmanMax
        else:
            return ghostSum / len(legalActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    #get current score
    score = currentGameState.getScore()
    #a big number for indicating ghost dangers
    indicator = 9999999
    
    if currentGameState.isWin() or currentGameState.isLose():
        return score

    #get pacman's position and the states of the ghosts
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    #get food locations as a list
    #and capsule locations as well
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    #for each ghost, find the distance between pacman and ghost
    #if ghost is close, act in response
    #if we can confidently say the ghost will be scared for long enough,
    #go get the ghost
    #otherwise do not move towards the ghost
    for ghost in ghostStates:
        dist = manhattanDistance(pos,ghost.getPosition())
        if ghost.scaredTimer > dist and dist < 3:
            return indicator
        elif ghost.scaredTimer < dist and dist < 3:
            return -indicator
    
    #find closest food and use the reciprocal to evaluate later
    foodDistance = findClosest(currentGameState,food,pos)
    foodScore = 1/foodDistance

    #find closest capsule (if any) and use reciprocal to evaluate later
    capsuleDistance = findClosest(currentGameState,capsules,pos)
    if capsuleDistance == 0:
        capsuleScore = 0
    else:
        capsuleScore = 1/capsuleDistance
    
    #played around with these numbers for a bit to see what was most effective
    #(as of right now, that would be: nothing)
    return 10*foodScore + 5*score + 10*capsuleScore


def findClosest(currentGameState,items, pacPos):
    #this does a bfs to find the closest item to pacman
    queue = util.Queue()
    queue.push(currentGameState)

    closed = set()

    while not queue.isEmpty():
        state = queue.pop()
        pos = x,y = state.getPacmanPosition()
        #if the list of item locations contains this future pacman position,
        #find the maze distance to that position from the current state
        #and return that as the distance
        if pos in items:
            distance = mazeDistance(pacPos,pos,currentGameState)
            return distance
        #otherwise, add the next depth to the queue
        if pos not in closed:
            closed.add(pos)
            for action in state.getLegalPacmanActions():
                nextState = state.generatePacmanSuccessor(action)
                queue.push(nextState)
        
        

# Abbreviation
better = betterEvaluationFunction
