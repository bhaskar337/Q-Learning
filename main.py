import random

class Q_learner:

	def __init__(self, R):
		#reward
		self.R = R
		#Q function
		self.Q = [[0]*6 for _ in range(6)]
		#future discount
		self.gamma = 0.8
		#final state
		self.goal = 5

		# exploration rate
		self.epsilon = 1.0 
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999

		
	#displays the Q matrix		
	def display_Q(self):
		for i in self.Q:
			for j in i:
				print(j, end='\t')
			print()	
		print()


	#gets all the possible actions of a state
	def get_actions(self, state):
		poss_actions = []
		for i, reward in enumerate(self.R[state]):
			if (reward != -1):
				poss_actions.append(i)

		return poss_actions

	#selects the optimal action from the Q matrix
	def optimal_action(self, state, poss_actions):
		max_a = 0
		actions = []
		for i in poss_actions:
			temp = self.Q[state][i]
			if temp == max_a:
				actions.append(i)
			elif temp > max_a:
				actions = [i]
				max_a = temp
		return random.choice(actions)


	#with probabity epsilon, does a random action
	#else selects optimal action
	def act(self, state, poss_actions):
		if random.random() <= self.epsilon:
		    return random.choice(poss_actions)
		return self.optimal_action(state, poss_actions)


	#converts Q matrix to percentage
	#divides each value in Q with max Q value 
	def to_percent(self):
		max_Q = 0
		for i in self.Q:
			temp = max(i)
			if (temp > max_Q):
				max_Q = temp
		for i, _ in enumerate(self.Q):
			self.Q[i][:] = [round(100 * x / max_Q) for x in self.Q[i]]


	#finds the optimal Q matrix
	def train(self, iterations=2000):
		#initial state 
		state = random.randint(0, 5)

		for _ in range(iterations):
			#all the possible actions the state has
			poss_actions = self.get_actions(state)

			#selecting an action
			action = self.act(state, poss_actions)

			#perform the action, i.e. going to the new state
			new_state = action

			#updating the Q matrix
			self.Q[state][action] = self.R[state][action] + self.gamma * max(self.Q[new_state])

			#decaying epsilon
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

			#if goal has been reached, we stop
			if new_state == self.goal:
				#select a new random initial state
				state = random.randint(0, 5)
				continue

			#else we change state and continue
			state = new_state

		#convert each value in Q matrix to percent
		self.to_percent()


	#finds optimal path from given state to goal
	def find_path(self, initial_state):
		state = initial_state
		print(state, end=' -> ')
		while True:
			poss_actions = self.get_actions(state)
			action = self.optimal_action(state, poss_actions)
			state = action
			if state == self.goal:
				print(state)
				break

			print(state, end=' -> ')

def main():
	#reward
	R = [
			[-1, -1, -1, -1, 0, -1],
			[-1, -1, -1, 0, -1 , 100],
			[-1, -1, -1, 0, -1, -1],
			[-1, 0, 0, -1, 0, -1],
			[0, -1, -1, 0, -1, 100],
			[-1, 0, -1, -1, 0, 100]
	]

	env = Q_learner(R)
	env.train()

	print('\nFinal Q matrix:')
	env.display_Q()

	env.find_path(2)

if __name__ == '__main__':
	main()

