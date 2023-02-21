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
        # "indicator" is a very big number
        # it tells pacman whether something is a very good idea or a very bad idea
        # specifically, it is used when ghosts are close or food is close to prioritize food collection
        # & discourage ghost contact
        indicator = 9999999

        # find ghosts
        ghostPositions = currentGameState.getGhostPositions()

        # sound the alarm and tell pacman that if this move puts him within 1 space of a ghost, DO NOT
        for ghostPosition in ghostPositions:
            if manhattanDistance(newPos, ghostPosition) < 2: 
                return -indicator
        
        numFood = currentGameState.getNumFood()
        nextNumFood = successorGameState.getNumFood()

        # cheer very loudly and tell pacman that if this move results in eating a pellet, DO IT
        if nextNumFood < numFood:
            return indicator

        
        # finding the size of the map
        width = newFood.width
        height = newFood.height

        # default value (longest distance possible on map)
        minDistance = manhattanDistance((0,0),(width,height))

        # find the closest food dot to pacman's newest position
        for i in range(0,width):
            for j in range(0,height):
                if newFood[i][j]:
                    distance = manhattanDistance(newPos,(i,j))
                    minDistance = min(minDistance, distance)

        # shorter distance = larger number = more incentive to move there
        return 1/minDistance

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
        # Run Minimiax algorithm with initial values:
        # current agent = pacman
        # current depth = 0
        _, action = self.minimax(gameState, 0, 0)
        return action
    
    def minimax(self, gameState, agent, currentDepth):
        """
        Recursive function that iterates through all legal actions to search for the optimal action.
        If the given agent is a MAX node it will find the action that maximizes reward.
        If the given agent is a MIN node it will find the action that minimizes reward.
        """
        # If reached the 'leaf' node or the max search depth then return evaluation of the current state
        if currentDepth == self.depth or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState), None
        
        # Find the next Agent
        nextAgent = (agent + 1) % gameState.getNumAgents()
        # Update depth counter if we cycle back to the pacman agent
        if nextAgent == 0:
            currentDepth += 1

        if agent == 0:
            # Pacman (ie. MAX agent)
            optValue = float('-inf')
            optAction = None
            # Iterate over all possible leagal actions and find the action that results in MAXIMUM reward
            for action in gameState.getLegalActions(agent):
                nextValue,_ = self.minimax(gameState.generateSuccessor(agent, action), nextAgent, currentDepth)
                if nextValue > optValue:
                    optValue = nextValue
                    optAction = action
        else:
            # Ghosts (ie. MIN agents)
            optValue = float('inf')
            optAction = None
            # Iterate over all possible leagal actions and find the action that results in MINIMUM reward
            for action in gameState.getLegalActions(agent):
                nextValue,_ = self.minimax(gameState.generateSuccessor(agent, action), nextAgent, currentDepth)
                if nextValue < optValue:
                    optValue = nextValue
                    optAction = action
        return optValue, optAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Run alphabeta pruning with initializing the following:
        # current agent = pacman
        # current depth = 0
        # alpha         = -inf
        # beta          = inf
        _, action = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return action
    
    def alphaBeta(self, gameState, agent, currentDepth, a, b):
        """
        Recursive function that iterates through all legal actions to search for the optimal action.
        If the given agent is a MAX node it will find the action that maximizes reward and it will
        update the alpha value and prune if the given optimal value > beta.
        If the given agent is a MIN node it will find the action that minimizes reward and it will
        update the beta value and prune if the given optimal value < alpha.
        """
        # If reached the 'leaf' node or the max search depth then return evaluation of the current state
        if currentDepth == self.depth or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState), None
        
        # Find the next Agent
        nextAgent = (agent + 1) % gameState.getNumAgents()
        # Update depth counter if we cycle back to the pacman agent
        if nextAgent == 0:
            currentDepth += 1

        if agent == 0:
            # Pacman (ie. MAX agent)
            optValue = float('-inf')
            optAction = None
            # Iterate over all possible leagal actions and find the action that results in MAXIMUM reward
            for action in gameState.getLegalActions(agent):
                nextValue, _ = self.alphaBeta(gameState.generateSuccessor(agent, action), nextAgent, currentDepth, a, b)
                if nextValue > optValue:
                    optValue = nextValue
                    optAction = action
                    # MAX node alpha-beta pruning
                    if optValue > b:
                        return optValue, optAction
                    a = max(a, optValue)
            return optValue, optAction
        else:
            # Ghosts (ie. MIN agents)
            optValue = float('inf')
            optAction = None
            # Iterate over all possible leagal actions and find the action that results in MAXIMUM reward
            for action in gameState.getLegalActions(agent):
                nextValue, _ = self.alphaBeta(gameState.generateSuccessor(agent, action), nextAgent, currentDepth, a, b)
                if nextValue < optValue:
                    optValue = nextValue
                    optAction = action
                    # MIN node alpha-beta pruning
                    if optValue < a:
                        return optValue, optAction
                    b = min(b, optValue)
            return optValue, optAction
    
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
        
        # Identify max score using recursion to search tree
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

    # Method to search tree for expected score
    def getExpectedScore(self, gameState, agentIndex: int, depth: int):
        
        #no search needed if nothing left to search
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Identify legal actions and the next agent to make a move (rollover to 0
        # If all ghosts have taken their turn)
        legalActions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        if nextAgent > gameState.getNumAgents() - 1:
            nextAgent = 0
        
        # If rolling over to 0, we must be at the next depth of game states
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        pacmanMax = 0
        ghostSum = 0

        # For every possible action, generate a successor, find the score by
        # Recursively calling this method
        # If pacman, take the maximum score from successors
        # If ghost, sum up scores to return an average (since equal probability)
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
    
    # Get current score
    score = currentGameState.getScore()
    # Big number for indicating ghost dangers
    indicator = 9999999
    
    if currentGameState.isWin() or currentGameState.isLose():
        return score

    # Get pacman's position and the states of the ghosts
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    # Get food and capsule locations as a list
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    # For each ghost, find the Manhattan distance between pacman and ghost
    # if ghost is close, act in response
    # if we can confidently say the ghost will be scared for long enough,
    # go get the ghost
    # otherwise do not move towards the ghost
    for ghost in ghostStates:
        dist = manhattanDistance(pos,ghost.getPosition())
        if ghost.scaredTimer > dist and dist < 3:
            return indicator
        elif ghost.scaredTimer < dist and dist < 3:
            return -indicator
    
    # Find closest food and use the reciprocal to evaluate later
    foodDistance = findClosest(currentGameState,food,pos)
    foodScore = 1/foodDistance

    # Find closest capsule (if any) and use reciprocal to evaluate later
    capsuleDistance = findClosest(currentGameState,capsules,pos)
    if capsuleDistance == 0:
        capsuleScore = 0
    else:
        capsuleScore = 1/capsuleDistance
    
    # Played around with these numbers for a bit to see what was most effective
    # (as of right now, that would be: nothing)
    return 10 * foodScore + 1 * score + 10 * capsuleScore


def findClosest(currentGameState, items, pacPos):
    # Calculate distance to each food positions available.
    distances = []
    for foodPos in items:
        distances.append(mazeDistance(pacPos, foodPos, currentGameState))
    
    # Return Minimum distance to a food
    if len(distances) > 0:
        return min(distances)
    else:
        return 0
        

# Abbreviation
better = betterEvaluationFunction
