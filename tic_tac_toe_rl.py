import numpy as np 
import matplotlib.pyplot as plt

SIZE = 3

np.random.seed(0)

class Agent:
	def __init__(self, eps=0.1, lr=0.5):
		self.eps = eps
		self.lr = lr
		self.state_hist = []
		self.winner_hist = []
		self.print_value_fn = False

	def set_print_value_fn(self, v):
		self.print_value_fn = v

	def set_eps(self, v):
		self.eps = v

	def set_value_fn(self, value_fn):
		self.value_fn = value_fn

	def set_symbol(self, symbol):
		self.symbol = symbol

	def update_winner_hist(self, s):
		self.winner_hist.append(s)

	def update_state_hist(self, s):
		self.state_hist.append(s)

	def reset_state_hist(self):
		self.state_hist = []

	def update_value_fn(self, env):
		reward = env.get_reward(self.symbol)
		target = reward

		for idx in reversed(self.state_hist):
			value = self.value_fn[idx] + self.lr * (target - self.value_fn[idx])
			
			self.value_fn[idx] = value 
			target = value 

		self.reset_state_hist()

	def take_action(self, env):
		# Epsilon Greedy
		rand = np.random.random()
		if rand < self.eps:
			# Random action
			possible_move = []
			for i in range(SIZE):
				for j in range(SIZE):
					if env.is_empty(i, j):
						possible_move.append((i, j))

			idx = np.random.choice(len(possible_move))
			next_move = possible_move[idx]

			if self.print_value_fn:
				print('Take random action!')
		else:
			best_state = None
			best_value = -999
			pos2value = {} 
			# action based on value function
			for i in range(SIZE):
				for j in range(SIZE):
					if env.is_empty(i, j):
						# Make a move
						env.board[i, j] = self.symbol
						# Get state
						state = env.get_state()
						state_value = self.value_fn[state]
						pos2value[(i,j)] = state_value
						# Check best state value
						if state_value > best_value:
							best_value = state_value
							best_state = state
							next_move = (i, j)

						# Reset the move
						env.board[i, j] = 0

			if self.print_value_fn:
				print('Value Function (Agent) :')
				for i in range(SIZE):
					print('----------------------')
					for j in range(SIZE):
						print('| ', end='')
						if env.is_empty(i, j):
							print("%.2f " % pos2value[(i,j)], end="")
						else:
							if env.board[i, j] == env.x:
								print(' x   ', end='')
							elif env.board[i, j] == env.o:
								print(' o   ', end='')
							else:
								print('    ', end='')

					print('|')
				print('----------------------')

		env.board[next_move[0], next_move[1]] = self.symbol

class Human:
	def __init__(self):
		pass

	def set_value_fn(self, value_fn):
		pass

	def set_symbol(self, symbol):
		self.symbol = symbol

	def update_winner_hist(self, s):
		pass

	def update_state_hist(self, s):
		pass

	def reset_state_hist(self):
		pass

	def update_value_fn(self, env):
		pass

	def take_action(self, env):
		while True:
			next_move = input('Enter coordinates i, j for your next move (e.g, 0,1 ): ')
			i, j = next_move.split(',')
			i = int(i)
			j = int(j)

			if env.is_empty(i,j):
				env.board[i, j] = self.symbol
				break

class Environment:
	def __init__(self):
		self.board = np.zeros((SIZE, SIZE))
		self.x = 1
		self.o = -1
		self.winner = None
		self.ended = False
		self.n_states = 3**(SIZE*SIZE)

	def is_empty(self, i, j):
		return self.board[i, j] == 0

	def get_state(self):
		k = 0
		h = 0
		for i in range(SIZE):
			for j in range(SIZE):
				if self.board[i, j] == 0:
					v = 0
				elif self.board[i, j] == self.x:
					v = 1
				elif self.board[i, j] == self.o:
					v = 2

				h += (3**k) * v
				k += 1

		return h

	def get_reward(self, symbol):
		if not self.is_game_over():
			return 0

		return 1 if self.winner == symbol else 0

	def is_game_over(self):
		# Return True if any winner or draw 
		# Check rows
		for i in range(SIZE):
			for player in [self.x, self.o]:
				if self.board[i].sum() == player * SIZE:
					self.winner = player
					self.ended = True
					return True

		# Check columns
		for j in range(SIZE):
			for player in [self.x, self.o]:
				if self.board[:, j].sum() == player * SIZE:
					self.winner = player
					self.ended = True
					return True

		# Check diagonals
		# Top left --> bottom right
		for player in [self.x, self.o]:
			if self.board.trace() == player * SIZE:
				self.winner = player
				self.ended = True
				return True

		# Top right --> bottom left
		for player in [self.x, self.o]:
			if np.flip(self.board, 1).trace() == player * SIZE:
				self.winner = player
				self.ended = True
				return True

		# Check if draw
		if np.abs(self.board).sum() == SIZE*SIZE:
			self.winner = None
			self.ended = True
			return True

		# Game is not over
		self.winner = None
		self.ended = False
		return False

	def print_board(self):
		print('============ Board ============')
		for i in range(SIZE):
			print('-------------')
			for j in range(SIZE):
				print('| ', end='')
				if self.board[i, j] == self.x:
					print('x ', end='')
				elif self.board[i, j] == self.o:
					print('o ', end='')
				else:
					print('  ', end='')
			print('|')
		print('-------------')

def play_game(p1, p2, env, print_board=False):
	current_player = None

	if print_board:
		env.print_board()

	while not env.is_game_over():

		# Player1 always starts first
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1

		# Take action
		current_player.take_action(env)

		if print_board:
			env.print_board()

		# Update state history
		state = env.get_state()
		p1.update_state_hist(state)
		p2.update_state_hist(state)

	# Update value function
	p1.update_value_fn(env)
	p2.update_value_fn(env)

	# Update winner history
	if env.winner == env.x:
		p1.update_winner_hist(1)
		p2.update_winner_hist(0)

	elif env.winner == env.o:
		p1.update_winner_hist(0)
		p2.update_winner_hist(1)

	else:
		p1.update_winner_hist(0)
		p2.update_winner_hist(0)

	if print_board:
		if env.winner == env.x:
			print('Game Over -> Winner is X')
		elif env.winner == env.o:
			print('Game Over -> Winner is O')
		else:
			print('Game Over -> Draw')

def convert_base10_to_base3 (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

def get_hash_state_winner_ended(env):
	results = []
	for i in range(env.n_states):
		base3 = convert_base10_to_base3(i)
		
		board = np.zeros(9)
		for i, s in enumerate(base3):
			if s == '0':
				board[i] = 0
			elif s == '1':
				board[i] = env.x
			elif s == '2':
				board[i] = env.o
		
		env.board = board.reshape(SIZE, SIZE)
		state = env.get_state()
		ended = env.is_game_over()
		winner = env.winner

		results.append((state, winner, ended))

	return results

def initial_v_x(env, state_winner_eneded_list):
	value_fn = np.zeros(env.n_states)

	for state, winner, endded in state_winner_eneded_list:
		if endded:
			if winner == env.x:
				v = 1
			else:
				v = 0
		else:
			v = 0.5

		value_fn[state] = v

	return value_fn

def initial_v_o(env, state_winner_eneded_list):
	value_fn = np.zeros(env.n_states)

	for state, winner, endded in state_winner_eneded_list:
		if endded:
			if winner == env.o:
				v = 1
			else:
				v = 0
		else:
			v = 0.5

		value_fn[state] = v

	return value_fn

if __name__ == '__main__':

	p1 = Agent(eps=0.1, lr=0.1)
	p2 = Agent(eps=0.1, lr=0.1)

	env = Environment()
	state_winner_eneded_list = get_hash_state_winner_ended(env)

	x_value_fn = initial_v_x(env, state_winner_eneded_list) 
	o_value_fn = initial_v_o(env, state_winner_eneded_list) 

	p1.set_value_fn(x_value_fn)
	p2.set_value_fn(o_value_fn)

	p1.set_symbol(env.x)
	p2.set_symbol(env.o)

	T = 20000
	for i in range(T):
		env = Environment()
		play_game(p1, p2, env)

		if i%1000 == 0:
			print(i)

	p1_winner_hist = np.array(p1.winner_hist)
	p2_winner_hist = np.array(p2.winner_hist)

	N = len(p1_winner_hist)

	p1_cumulative_avg = np.cumsum(p1_winner_hist) / (np.arange(N) + 1)
	p2_cumulative_avg = np.cumsum(p2_winner_hist) / (np.arange(N) + 1)

	plt.plot(p1_cumulative_avg)
	plt.plot(p2_cumulative_avg)
	plt.xscale('log')
	plt.legend(['Agent 1', 'Agent 2'])
	plt.title('Cumulative Win Rate')
	plt.xlabel('Episode')
	plt.show()

	human = Human()

############### Agent 1 vs Human ##################
	human.set_symbol(env.o)
	p1.set_print_value_fn(True)
	p1.set_eps(0)

	while True:
		env = Environment()
		play_game(p1, human, env, print_board=True)
		answer = input("Play again? [y/n]: ")
		if answer  == 'n':
		  break

############### Human vs Agent 2 ##################
	# human.set_symbol(env.x)
	# p2.set_print_value_fn(True)
	# p2.set_eps(0)

	# while True:
	# 	env = Environment()
	# 	play_game(human, p2, env, print_board=True)
	# 	answer = input("Play again? [y/n]: ")
	# 	if answer  == 'n':
	# 	  break

############### Human vs  Human ##################
	# p1 = Human()
	# p2 = Human()
	# env = Environment()

	# p1.set_symbol(env.x)
	# p2.set_symbol(env.o)

	# while True:
	# 	play_game(p1, p2, env, print_board=True)

	# 	answer = input("Play again? [y/n]: ")
	# 	if answer  == 'n':
	# 	  break


