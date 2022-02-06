
import numpy as np

class obsToPlanes():
    def __init__(self, size, max_bombs=5, max_flame_life=3):
        # 18 x lenght x width planes
        self.planes = np.zeros((18, size, size))
        self.size = size
        self.max_bomb_count = max_bombs
        self.max_flame_life = max_flame_life

    def concatObservations(self, last_obs, current_obs, message_obs):
        return np.concatenate((last_obs, current_obs, message_obs), axis=2)

    def planeFilling(self, obs, planes, max_bombs_agent=1):
        
        """
        obs == state dictionary of an agent

        Observation Space

        Planes

        Obstacles
        * Non-Destructible 0
        * Destructible 1

        Items
        * Increase Bomb Count --> onehot 2
        * Increase Bomb Strength (Range) --> onehot 3
        * Kick --> onehot 4

        Bomb:
        * Bomb Position & Life 0 -> 1 --> procentual 5
        * Bomb Blast Strength --> strength / boardsize 6

        Bomb Movement:
        * Bomb Horizontal movement left(3) = 1, right(4) = -1    7
        * Bomb Vertical movement up = 1(1), down(2) = -1         8

        Flames:
        * Bomb Flame Position & Life 1 -> 0 --> procentual 9

        Player
        * Position Self 10
        * Position Enemy 1 11
        * Position Enemy 2 12
        * Position Enemy 3 13

        Scalar Feature Planes:
        * Self: Player Bomb Strength --> strenght / boardsize 14
        * Self: Bomb Count (Ammo) --> float 15 bomb count / Max_bomb_count == 5
        * Self: Max Bomb Count --> float 16 max bomb count / Max_bomb_count == 5
        * Self: Can Kick --> int 17

        """
        for row in range(self.size):
            for col in range(self.size):
                planes = self.cellToBoardPlanes(obs, planes, row, col, max_bombs_agent)
        return planes

    
    def cellToBoardPlanes(self, obs, planes, row, col, max_bombs_agent):
        index = obs["board"][row, col]
        if index != 0 and index != 5:
            plane, val = self.ValueToPlaneAndValue(index)
            if val == -1:  # Bomb or Flame
                if plane == 5:  # Bomb
                    val = 1 / obs["bomb_life"][row, col]
                    planes[6, row, col] = obs["bomb_blast_strength"][row, col] / self.size
                    if obs["bomb_moving_direction"][row, col] != 0:
                        move_plane, move_value = self.bombToMovement(obs["bomb_moving_direction"][row, col]) 
                        planes[move_plane, row, col] = move_value
                elif plane == 9:  # Flame
                    val = 1 - ( 1 - (obs["flame_life"][row, col] / self.max_flame_life))
                else:
                    raise ValueError("no bomb and no flame")
            planes[plane, row, col] = val
        

        # Other information to planes
        planes[14, row, col] = obs["blast_strength"] / self.size
        planes[15, row, col] = obs["ammo"] / self.max_bomb_count if obs["ammo"] != 0 else 0
        planes[16, row, col] = max_bombs_agent / self.max_bomb_count
        planes[17, row, col] = 1 if obs["can_kick"] else 0
        return planes

    def ValueToPlaneAndValue(self, value):
        """
        translates the given value into some the 18 planes
        returns (plane_index, value)
        """
        if value == 1: 
            return 0, 1
        elif value == 2: 
            return 1, 1
        elif value == 3: 
            return 5, -1
        elif value == 4:
            return 9, -1
        elif value == 6: 
            return 2, 1
        elif value == 7: 
            return 3, 1
        elif value == 8: 
            return 4, 1
        elif value == 10: 
            return 10, 1 # TODO check how self and team and etc is done
        elif value == 11: 
            return 11, 1
        elif value == 12: 
            return 12, 1
        elif value == 13: 
            return 13, 1
        else:
            raise ValueError("Error in value of obs field")
    
    def bombToMovement(self, direction):
         # returns plane_index, value
        if direction == 1: return 8, 1
        elif direction == 2: return 8, -1
        elif direction == 3: return 7, 1
        elif direction == 4: return 7, -1
        else: raise ValueError("something wrong with bomb movement")

def main():
    count = 0
    data = np.load("./states.npy", allow_pickle=True)
    obj = obsToPlanes(11)
    for game in data:
        for state in game:
            for agent in state:
                planes = obj.planeFilling(agent, obj.planes)  
        count += 1
        print(count)
        if count > 5: break


if __name__ == "__main__":
    main()

