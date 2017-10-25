"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Qlearner implementation by Wenchen Li

Qlearner:

dyna-Q:
we start with straight, regular
Q-Learning here and then we add three new components.
The three components are, we update models of T and R, then we hallucinate an experience
and update our Q table.
Now we may repeat this many times, in fact maybe hundreds of times, until we're happy.
Usually, it's 1 or 200 here.
Once we've completed those, we then return back up to the top and continue our interaction
with the real world.
The reason Dyna-Q is useful is that these experiences with the real world are potentially very
expensive and these hallucinations can be very cheap. And when we iterate doing many of
them, we update our Q table much more quickly.
"""

import numpy as np
import random as rand
from util import save, load

class QLearner(object):
  """
  The constructor QLearner() should reserve space for keeping track of Q[s, a] for the number of states and actions. It initialize Q[] with all zeros.

  num_states integer, the number of states to consider
  num_actions integer, the number of actions available.
  alpha float, the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
  gamma float, the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
  rar float, random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
  radr float, random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
  dyna integer, conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
  verbose boolean, if True, your class is allowed to print debugging statements, if False, all printing is prohibited.
  """

  def __init__(self,
               num_states,
               num_actions,
               alpha=0.2,
               alpha_decay=.9999,
               gamma=0.9,
               rar=0.5,
               radr=0.99,
               dyna=0,
               verbose=False):

    self.verbose = verbose
    self.num_actions = num_actions
    self.num_states = num_states
    self.s = 0
    self.a = 0
    self.q_table = np.zeros((num_states,num_actions))
    self.rar = rar
    self.radr= radr
    self.state_rar = np.zeros(num_states)
    self.state_rar+=rar
    self.alpha = alpha
    self.alpha_decay = alpha_decay
    self.gamma = gamma

    self.dyna = dyna
    self.dyna_lr = .8
    self.dyna_init = dyna
    self.t_table_increment_unit = 1.0 / 100
    self.t_table = np.ones((num_actions*num_states,num_states))* self.t_table_increment_unit# state transition,
    self.r_table = np.zeros((num_states,num_actions))


    self.last_state = rand.randint(0, self.num_states - 1)
    self.last_action = None

  def decay_alpha(self):
    """
    decay the learning rate alpha in the value iteration
    """
    self.alpha *=self.alpha_decay

  def querysetstate(self, s):
    """
        @summary: Update the state without updating the Q-table
        @detail: A special version of the query method that sets the state to s,
        and returns an integer action according to the same rules as query()
        (including choosing a random action sometimes), but it does not execute
        an update to the Q-table. It also does not update rar. There are two main
        uses for this method: 1) To set the initial state, and 2) when using a
        learned policy, but not updating it.
        @param s:int, The new state
        @returns:int, The selected action
        """
    # self.s = s
    # action = rand.randint(0, self.num_actions - 1)
    # if self.verbose: print "s =", s, "a =", action
    # return action
    # exploration vs exploitation
    # if rand.uniform(0, 1) <= self.state_rar[self.last_state]:  # exploration
    #   action = rand.randint(0, self.num_actions - 1)
    #   self.state_rar[self.last_state] *= self.radr
    #   # self.rar *= self.radr  # exploration decay update
    # exploitation
    action = np.argmax(self.q_table[s])
    assert action < self.num_actions
    # self.q_table[self.last_state][self.last_action] += r

    if self.verbose: print "s =", s, "a ="
    # update state and action
    self.last_state = s
    self.last_action = action
    return action

  def query(self, s_prime, r):
    """

        @summary: Update the Q table and return an action
        @detail: the core method of the Q-Learner. It should keep track
        of the last state s and the last action a, then use the new information
        s_prime and r to update the Q table. The learning instance, or experience
        tuple is <s, a, s_prime, r>. query() should return an integer, which is
        the next action to take. Note that it should choose a random action with
        probability rar, and that it should update rar according to the decay
        rate radr at each step. Details on the arguments:
        @param s_prime: int,  the new state.
        @param r :float, a real valued immediate reward.
        @returns:int, The selected action
        """
    # learning_instance = [self.last_state,self.last_action,s_prime,r]

    # exploration vs exploitation
    if rand.uniform(0, 1) <= self.state_rar[self.last_state]:  # exploration
      action = rand.randint(0, self.num_actions - 1)
      self.state_rar[self.last_state] *= self.radr
    else:# exploitation
      action = np.argmax(self.q_table[s_prime])
      assert action < self.num_actions
      self.q_table[self.last_state][self.last_action] = (1.0-self.alpha)*self.q_table[self.last_state][self.last_action] + self.alpha* (r + self.gamma*self.q_table[s_prime][np.argmax(self.q_table[s_prime])]) #bellman

    if self.verbose: print "s =", s_prime, "a =", action, "r =", r

    # update state and action in Qlearn
    self.last_state = s_prime
    self.last_action = action

    # add dyna
    ## dyna: update T and R table
    while self.dyna > 0:
      # update T
      transition_index = self.num_states * self.last_action + self.last_state
      self.t_table[transition_index][s_prime] += self.t_table_increment_unit

      # update R
      self.r_table[self.last_state][self.last_action] = (1-self.dyna_lr) * self.r_table[self.last_state][self.last_action] + self.dyna_lr * r

    ## dyna hallucinate
      dyna_state = rand.randint(0, self.num_states - 1)
      dyna_action = rand.randint(0, self.num_actions - 1)
      transition_index = self.num_states * dyna_action + dyna_state
      transition_prob = self.t_table[transition_index] / np.sum(self.t_table[transition_index])
      state_infer_from_t_table = np.random.choice(range(self.num_states), 1, p=transition_prob)[0]
      dyna_r = self.r_table[dyna_state][dyna_action]

    ## dyna update Q table
      self.q_table[dyna_state][dyna_action] = (1.0 - self.alpha) * self.q_table[dyna_state][dyna_action] + self.alpha * (dyna_r + self.gamma * self.q_table[state_infer_from_t_table][np.argmax(self.q_table[state_infer_from_t_table])])
      self.dyna-=1
    # end of dyna
    self.dyna =  self.dyna_init

    return action

  def save_model(self,table_name="q_learner_tables.pkl"):
    tables = [self.q_table,self.t_table,self.r_table]
    save(tables,table_name)

  def load_model(self,table_name="q_learner_tables.pkl"):
    [self.q_table, self.t_table, self.r_table] = load(table_name)

if __name__ == "__main__":
  print "Remember Q from Star Trek? Well, this isn't him"
