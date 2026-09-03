# agent.py
import random
from collections import deque
import heapq
import math


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    
    def manhattan_distance(self, pos, goal):
        return int(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def euclidean_distance(self, pos, goal):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])
   

    # Map a movement action to its (dx, dy) offset. Matches execute_action() in
    # visual_grid_game.py: Up = +y, Down = -y, Left = -x, Right = +x.
    MOVES = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        # Step 1.3: the offline plan and the active search strategy.
        self.plan = []
        self.active_algo = 'BFS'   # Change to 'DFS' or 'UCS' to compare behaviour.

    # ------------------------------------------------------------------ helpers
    def _get_successors(self, state, grid_size, walls):
        """Yield (action, next_state) pairs that are on-grid and not walls."""
        width, height = grid_size
        x, y = state
        for action, (dx, dy) in self.MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _closest_food(self, start, all_food):
        """Pick the food pellet with the smallest Manhattan distance from start."""
        return min(
            all_food,
            key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1])
        )

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """
        A* search combining g(n) and h(n): f(n) = g(n) + h(n).
        start_pos and goal_pos are (x, y) tuples. walls is a set of blocked coordinates.
        heuristic_type: 'manhattan' or 'euclidean'
        Returns a list of actions (path) to reach goal_pos, or [] if none found.
        """
        # helper to select heuristic
        if heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        frontier = []  # heap of (f_cost, g_cost, current_pos, path_taken)
        g0 = 0
        h0 = heuristic(start_pos, goal_pos)
        f0 = g0 + h0
        heapq.heappush(frontier, (f0, g0, start_pos, []))
        reached_states = set()
        width, height = grid_size

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            # goal test
            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            x, y = current_pos
            for action, (dx, dy) in self.MOVES.items():
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                # bounds and wall checks
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbor in walls:
                    continue
                if neighbor in reached_states:
                    continue

                g_new = g_cost + 1
                h_new = heuristic(neighbor, goal_pos)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, g_new, neighbor, path_taken + [action]))

        return []

    # ------------------------------------------------------------------ searches
    def bfs_search(self, start, goal, grid_size, walls):
        """Breadth-First Search: FIFO queue -> shallowest node first (optimal here)."""
        frontier = deque([(start, [])])          # (state, path_of_actions)
        reached = {start}                        # graph search: track explored states
        while frontier:
            state, path = frontier.popleft()     # FIFO
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []                                # no path found

    def dfs_search(self, start, goal, grid_size, walls):
        """Depth-First Search: LIFO stack -> deepest node first (winding paths)."""
        frontier = [(start, [])]                 # use a list as a stack
        reached = {start}
        while frontier:
            state, path = frontier.pop()         # LIFO
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []

    def ucs_search(self, start, goal, grid_size, walls):
        """Uniform-Cost Search: priority queue ordered by total path cost g(n)."""
        counter = 0                              # tie-breaker so tuples never compare paths
        frontier = [(0, counter, start, [])]     # (cost, tiebreak, state, path)
        reached = {start: 0}                     # best known cost to each state
        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                new_cost = cost + 1              # uniform step cost of 1 per move
                if nxt not in reached or new_cost < reached[nxt]:
                    reached[nxt] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nxt, path + [action]))
        return []

    # ------------------------------------------------------------------ act
    def sense_and_act(self, percept: dict) -> str:
        # Step 1.3: only plan when we have no plan left to execute.
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return 'Up'                      # nothing to chase; harmless default

            start = tuple(percept['agent_pos'])
            grid_size = percept['grid_size']
            walls = set(tuple(w) for w in percept['walls'])
            goal = self._closest_food(start, all_food)

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'AStar':
                # use manhattan by default for A*
                self.plan = self.astar_search(start, goal, walls, grid_size, heuristic_type='manhattan')

            # If search failed (e.g. food walled off), fall back to a single step.
            if not self.plan:
                return random.choice(list(self.MOVES.keys()))

        # Step 1.3: return and consume the first action of the plan.
        return self.plan.pop(0)
    
    


if __name__ == "__main__":
    agent = SearchAgent()
    start = (0, 0)
    goal = (3, 4)
    print("Manhattan:", agent.manhattan_distance(start, goal))   # expected 7
    print("Euclidean:", agent.euclidean_distance(start, goal))   # expected 5.0
