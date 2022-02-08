from . import BaseAgent


class StonerAgent(BaseAgent):
    def __init__(self): super(StonerAgent, self).__init__()

    def act(self, obs, action_space):
        return 0  # random.randint(1,4) #0
