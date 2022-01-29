'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent
from .random_agent import SmartRandomAgent
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent
from .agent007 import Agent007
from .simple_agent_cautious_bomb import CautiousAgent
