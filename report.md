[//]: # (Image References)


# 1. Learning Algorithm

## Agent and DQN Algorithm
The agent uses deep Q-learning (DQN) as described in [this](http://files.davidqiu.com//research/nature14236.pdf) article. Therefore it has two neural networks as function approximators, one local and one target network. The principal algorithm is as follows:
* Select eps-greedy action
* Execute action and get new state and reward from environment
* Store transition in Replay Buffer
* Sample random Minibatch from Replay Buffer and compute TD-target value with target network
* Perform gradient descent from local network Q-value to new TD-target with respect to the local network parameters
* Do soft transition for target network towards local network
Where these steps are repeated for each step in each episode. Note that the epsilon value for choosing the action epsilon greedy is reduced linearly over episodes, starting with `eps_start = 1.0` and ending at `eps_end = 0.01` after `eps_nEpisodes = 1000`. 

Further agent hyperparameters are:

	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size for learning
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network
	
As optimizer `Adam` has been used.

## Neural Network
As the observation space of the environment is `state_size = 37` the input size of the neural network matches this size and as `action_size = 4` the output size of the neural network matches this as well. Between input and output are two linear hidden layers, both with size `hidden_layers = [37*3, 37*3]` and `relu` activation. 

# 2. Plot of Rewards
With the above described agent the environment has been solven in 1029 episodes. The development of average rewards as well with all scores over each episode are provided below.

	Episode 100	Average Score: 0.09
	Episode 200	Average Score: 0.67
	Episode 300	Average Score: 1.46
	Episode 400	Average Score: 2.69
	Episode 500	Average Score: 4.11
	Episode 600	Average Score: 6.14
	Episode 700	Average Score: 7.94
	Episode 800	Average Score: 9.54
	Episode 900	Average Score: 10.82
	Episode 1000	Average Score: 12.59
	Episode 1029	Average Score: 13.01
	Environment solved in 1029 episodes!	Average Score: 13.01

![Score over Episodes for DQN agent](data/score_over_rewards.png)

# 3. Ideas for Future Work

The submission has concrete future ideas for improving the agent's performance.