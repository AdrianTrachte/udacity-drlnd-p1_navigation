# 1. Learning Algorithm

## Agent and DQN Algorithm
The agent uses deep Q-learning (DQN) as described in [this](http://files.davidqiu.com//research/nature14236.pdf) article. Therefore it has two neural networks as function approximators, one local and one target network. The principal algorithm is as follows:
* Select eps-greedy action
* Execute action and get new state and reward from environment
* Store transition in Replay Buffer
* Sample random Minibatch from Replay Buffer and compute TD-target value with target network
* Perform gradient descent from local network Q-value to new TD-target with respect to the local network parameters
* Do soft transition for target network towards local network
Where these steps are repeated for each step in each episode. Note that the epsilon value for choosing the action epsilon greedy is reduced linearly over episodes, starting with `eps_start = 1.0` and ending at `eps_end = 0.01` after `eps_nEpisodes = 500`. 

Further agent hyperparameters are:

	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size for learning
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network
	
As optimizer `Adam` has been used.

## Neural Network
As the observation space of the environment is `state_size = 37` the input size of the neural network matches this size and as `action_size = 4` the output size of the neural network matches this as well. Between input and output are two linear hidden layers, both with size `hidden_layers = [37, 37]` and `relu` activation. 

# 2. Plot of Rewards
With the above described agent the environment has been solved in 551 episodes. The development of average rewards as well with all scores over each episode are provided below.

	Episode 100	Average Score: 0.31
	Episode 200	Average Score: 2.32
	Episode 300	Average Score: 4.46
	Episode 400	Average Score: 7.93
	Episode 500	Average Score: 11.21
	Episode 551	Average Score: 13.00
	Environment solved in 551 episodes!	Average Score: 13.00

![Score over Episodes for DQN agent](./data/score_over_episodes_report.png "Score over Episodes")

# 3. Ideas for Future Work
As the DQN algorithm overestimates the action values by taking always the maximum of noisy data, the algorithm can be extended to a Double DQN (DDQN). In DDQN when evaluating the best action value the process is splitted into selection of the best action and evaluation of the best action. So both have to agree on the best action, otherewise the resulting action value is not as high. For the evaluation part the already implemented target neural network can be reused. This [paper](https://arxiv.org/abs/1509.06461) gives the details about this concept and how DQN overestimates the action values.

Another thing worth investigating further is prioritized experience replay. The main idea behind this concept is, that not all experiences are equally important. Therefore to each experience / transition the TD-error is added as a measure of priority. With this a probability can be formulated, favoring experiences more important. Though some further extensions are necessary to assure a certain amount of randomness and the update rule has to incorporate this changed selection process as well. More details can be found in this [paper](https://arxiv.org/abs/1511.05952).
