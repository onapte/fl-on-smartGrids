# Markov Decision process

class state(object):
    def __init__(self, x, y, reward=-1, isReachable=True, isTerminalState=False):
        self.x = x
        self.y = y
        self.loc = (x, y)
        self.reward = reward
        self.isReachable = isReachable
        self.isTerminalState = isTerminalState
        self.isStartLocation = self.isStartLoc()
        self.freeLocs = [(0, 1), (1, 0)]

    def blockable(self):  # Return true if blockable. Terminal and freeLocs are not blockable
        return self.playable() and not self.loc in self.freeLocs

    def playable(self):  # playable states are Reachable non terminal states
        return self.isReachable and not self.isTerminalState

    def isStartLoc(self):  # Return true if at the starting location
        if self.loc == (0, 0):
            return True
        return False

    def setReward(self, reward):  # To set reward associated with each state
        self.reward = reward

    def isAccessible(self):  # Return true if the state is reachable
        return self.isreachable

    def block(self):  # Add a block to a particular state
        self.isReachable = False

    def getReward(self):  # Return reward for state
        return self.reward

    def setAsTerminal(self):  # Set a particular state as terminal
        self.isTerminalState = True

    def isTerminal(self):  # Return True if state is terminal
        return self.isTerminalState

state(1, 1, -1, True, False)

class Environment(object):
    """The environment is the interface between the agent and the grid world, it creates and consists of the grid 
    world as well as all underlying transition rules between states. It keeps track of the present state
    receives actions and generates feedback to the agent and controls transition between states
    properties:
        self.xDim, yDim, numBlocks ==> See designGridWorld function
        self.transitionProb: (float) between 0 and 1. defines the probability of moving to the desired
        location. It introduces stochasticity to the environment where the same action could produce 
        different reactions from the environment
        self.initState: (state) the starting position (0,0)
        self.actionDict: (dictionary) of all actions
    
    
    """
    def __init__(self, xDim, yDim, numBlocks, transitProb):
        self.xDim = xDim   
        self.yDim = yDim
        self.numBlocks = numBlocks
        self.transitProb = transitProb
        self.grid = designGridWorld(self.xDim, self.yDim, self.numBlocks)
        self.initState = self.grid[0,0]
        self.state = self.initState
        self.reward = 0
        self.action_dict = {0: "remained in place", 1: "Moved up", 2: "Moved down", 3: "Moved left", 
                      4: "Moved right "}
        
    
    def goalAchieved(self): #returns whether the goal has been reached
        return self.state == self.grid[-1,0]
        
    
    def move(self, action): #The movement produced by an action. 
        #The new transition is controlled by this parameter and it introduces uncertainity to the movement
        rand = np.random.rand()
        if rand <= self.transitProb:
            return  action 
        else:
            return np.random.randint(5)
    
    def reset(self): #Restart and set to the intial State
        self.state = self.initState
        print("Grid world reset")
        print("Position: ({}, {})".format(self.state.x, self.state.y))
    
    def nextStep(self, action): #The Rules following the agents selection of an action
        action = self.move(action) #The environment returns a stochastic map from the action to the movement 
        if action == 0:
            self.nextState = self.state #Remain in place
        if action == 1: #Move up if not at the top of the grid, else remain in place
            if self.state.x == 0:
                action =0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x-1, self.state.y] 
            
        elif action ==2: #Go down if not at the bottom , remain in place otherwise
            if self.state.x == self.xDim-1:
                print("bottom")
                action =0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x+1, self.state.y]
            
        elif action ==3: #If at the left border, remain in place, otherwise move left
            if self.state.y == 0:
                action =0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x, self.state.y-1]  
            
        elif action ==4: #If at the right border, remain in place, otherwise move right
            if self.state.y == self.yDim-1:
                action =0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x, self.state.y+1] 
        if not self.nextState.isReachable: #If the chosen state is blocked, remain in place
            action = 0
            print("oops, you hit an obstacle")
            self.nextState = self.state
        self.state = self.nextState #The next state becomes the present state
        print(self.action_dict[action])   
        print("New position: ({}, {})".format(self.state.x, self.state.y))
        return self.state