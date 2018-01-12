import sys; sys.path.insert(0, '..')

import gym
from gym import wrappers
from util import fully_connected
import tensorflow as tf
import numpy as np

class Agent(object):

    def __init__(self, learning_rate):

        self.observations = tf.placeholder(tf.float32, shape=[1, 4], name='observations')
        hidden_layer      = fully_connected(self.observations, 8, with_activation=True, activation=tf.nn.relu)
        probability       = fully_connected(hidden_layer, 1, with_activation=True, activation=tf.nn.sigmoid)
        complementary     = tf.subtract(1.0, probability)
        output            = tf.concat([probability, complementary], 1, name='action_probabilities')
        log_likelihoods   = tf.log(output)
        self.action       = tf.multinomial(log_likelihoods, num_samples=1)[0][0]
        log_likelihood    = log_likelihoods[:, tf.to_int32(self.action)]

        # Create optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Compute gradients, returns a list of gradient variable tuples
        grads_and_vars = optimizer.compute_gradients(log_likelihood)
        # Extract gradients and inverse the sign of gradients
        # (compute_gradients returns inverted gradients for minimization)
        self.gradients = [grad * -1 for (grad, _) in grads_and_vars]

        # Create placeholders for modified gradients
        self.grad_placeholders = []
        for i, gradient in enumerate(self.gradients):
            self.grad_placeholders.append(tf.placeholder(tf.float32, gradient.shape, name=f'gradient_{i}'))

        # Apply gradients
        tvars = tf.trainable_variables()
        self.training_step = optimizer.apply_gradients(zip(self.grad_placeholders, tvars))

    def get_action(self, observation, session):
        '''Retrieve an action for a given set of observations.'''
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        grad_dummies = [np.zeros(grad.shape) for grad in self.grad_placeholders]
        feed_dict = {self.observations: observation}
        feed_dict.update({p: grad for (p, grad) in zip(self.grad_placeholders, grad_dummies)})
        return session.run([self.action] + self.gradients, feed_dict=feed_dict)

    def train(self, discounted_gradients, session):
        '''Apply discounted gradients computed over one episode.'''
        feed_dict = {self.observations: np.zeros((1, 4), np.float32)}
        feed_dict.update({p: grad for (p, grad) in zip(self.grad_placeholders, discounted_gradients)})
        session.run([self.training_step], feed_dict=feed_dict)


def discounted_rewards(rewards_per_time, discount_factor):
    '''Calculate the discounted rewards over a time series of rewards.

    Parameters
    ----------
    rewards_per_time    :   list or np.ndarray
                            List of rewards observed for each time step
    Returns
    -------
    np.ndarray
        Discounted rewards

    '''
    if not isinstance(rewards_per_time, np.ndarray):
        rewards_per_time = np.array(rewards_per_time)

    T = len(rewards_per_time)
    # create vector with all elems set to factor and power it with [0,...,T-1]
    factors = np.power(np.full((T,), discount_factor, np.float32), np.arange(T))
    discounted_rewards = [np.dot(rewards_per_time[t:], factors[:T-t]) for t in range(T)]
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
    return discounted_rewards


def discounted_gradients(grads, discounted_rewards):
    '''Discount gradients by multiplying with rewards

    Parameters
    ----------
    grads   :   list
                Raw gradients
    discounted_rewards  :   list or np.ndarray
                            Reward factors as returned by `discounted_rewards()`

    Returns
    -------
    np.ndarray
        Discounted gradients

    '''
    discounted_gradients = []
    # each time step is a list of arrays, so multiply each of those with the time step's discount
    # factor'
    for t in range(len(grads)):
        discounted_gradients.append([g * discounted_rewards[t] for g in grads[t]])

    return discounted_gradients


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    episodes               = 2000
    iters                  = 200
    render_step            = 100
    summed_gradient_buffer = []

    agent = Agent(0.01)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for episode in range(episodes):
            gradient_buffer = []
            reward_buffer   = []
            observation     = env.reset()
            for i in range(iters):
                act, *grads = agent.get_action(observation, session)

                if episode > 500:
                    if episode * i % render_step == 0:
                        env.render()

                observation, reward, done, info = env.step(act)
                reward_buffer.append(reward)
                gradient_buffer.append(grads)

                if done:
                    print(f'Episode finished after {i+1} timesteps')
                    break

            ##################
            #  Episode done  #
            ##################
            # summed_gradient_buffer.append(np.sum(np.array(gradient_buffer, 0)))
            rewards = discounted_rewards(reward_buffer, 0.5)
            gradients = discounted_gradients(gradient_buffer, rewards)
            gradient_sum = np.sum(np.array(gradients), 0)
            agent.train(gradient_sum, session)


if __name__ == '__main__':
    main()
