import math
import numpy as np
import matplotlib.pyplot as plt

max_cars = 10
max_transfer = 3
cost_per_transfer = 2
mu_in1 = 2
mu_in2 = 2
mu_out1 = 3
mu_out2 = 3

#create state space, action space, and reward space
S = [(i,j) for i in range(max_cars+1) for j in range(max_cars+1)]
A = {}
for s in S:
    i, j = s
    A[(i,j)]=list(range(max(-j, -max_transfer, -(max_cars-i)), min(i, max_transfer, max_cars-j)+1))
R = list(range(-cost_per_transfer*max_transfer, max_cars**2+1))

def poisson(lamb, n):
    return (lamb**n)/math.factorial(n)*math.exp(-lamb)

def req_ret_probs(maximum, lamb):
    """Returns a list of the probabilities that n cars are returned or taken out
    based on Poisson with parameter lamb.  Maximum is the most cars that can be taken out or returned."""
    result = []
    for num_cars in range(maximum):
        prob = poisson(lamb, num_cars)
        result.append(prob) #result[num_cars] = prob
    result.append(1-sum(result)) #probability that 'maximum' cars are taken/returned
    return result

def create_P(S, A):
    """Creates the probability distribution P, a dictionary
    mapping 4-tuples (s', r, s, a) to probabilities p(s', r | s, a)"""
    P={}
    for s in S: # goes through all states
        for a in A[s]: #goes through all actions in that state
            s1, s2 = s #cars at beginning of day
            req1_probs = req_ret_probs(s1-a, mu_out1)
            req2_probs = req_ret_probs(s2+a, mu_out2)
            for req1 in range(len(req1_probs)): #goes through all pairs of requests
                for req2 in range(len(req2_probs)):
                    ret1_probs = req_ret_probs(max_cars-s1+a+req1, mu_in1)
                    ret2_probs = req_ret_probs(max_cars-s2-a+req2, mu_in2)
                    r = -2*abs(a)+10*req1+10*req2 #calculates reward
                    for ret1 in range(len(ret1_probs)): #goes through all pairs of returns
                        for ret2 in range(len(ret2_probs)):
                            s1_new = s1-a-req1+ret1
                            s2_new = s2+a-req2+ret2
                            s_new = (s1_new, s2_new)
                            #adds probability of this scenario to P
                            prob = ret1_probs[ret1]*ret2_probs[ret2]*req1_probs[req1]*req2_probs[req2]
                            if prob>0.0000000: #disregards low probability events (otherwise Memory Error)
                                P[(s_new, r, s, a)]=P.get((s_new, r, s, a), 0)+ prob
    return P

P = create_P(S, A)

def initialize(S, A):
    """S is the state space, a set of all possible states.
    A is a dictionary where the states are the keys
    and the key values are lists with actions that can be taken in that state.
    Returns a tuple containing a value function and a policy."""
    V = {}
    policy = {}
    for state in S:
        V[state]=0
        policy[state]=A[state][0]
    return V, policy

def policy_evaluation(P, S, R, policy, V, epsilon, gamma):
    """P is a dictionary mapping 4-tuples (s', r, s, a) to probabilities p(s', r | s, a).
    R is a list of possible rewards.
    Returns the modified value function V."""
    delta = 0
    while delta<=epsilon:
        delta = 0
        for state in S:
            old_value = V[state]
            V[state]=sum(P.get((s_prime, r, state, policy[state]), 0)*(r+gamma*V[s_prime])
                    for s_prime in S for r in R)
            delta = max(delta, abs(old_value-V[state]))
    return V

def policy_improvement(P, S, R, policy, V, gamma):
    """Improves the policy.  Modifies policy.  Returns the stability of the policy (boolean)"""
    policy_stable = True
    for state in S:
        old_action = policy[state]
        policy[state] = max(A[state], key=lambda a: sum(P.get((s_prime, r, state, a), 0)*(r+gamma*V[s_prime])
                    for s_prime in S for r in R))
        if not old_action == policy[state]:
            policy_stable=False
    return policy_stable

def policy_iteration(P, S, A, R, epsilon, gamma):
    print("Initializing.")
    V, policy = initialize(S,A)
    policy_stable = False
    iterations = 1
    while not policy_stable:
        print(grid)
        print("Evaluating policy", iterations)
        policy_evaluation(P, S, R, policy, V, epsilon, gamma)
        print("Improving policy", iterations)
        policy_stable = policy_improvement(P, S, R, policy, V, gamma)
        for i in range(max_cars+1):
            for j in range(max_cars+1):
                grid[i,j]=int(V[(i,j)])
        iterations+=1
    return policy

grid = np.zeros((max_cars+1, max_cars+1))
policy = policy_iteration(P, S, A, R, 0.001, 0.9)
print(grid)


"""
[[ 0. -1. -2. -3. -3. -3. -3. -3. -3. -3. -3.]
 [ 0. -1. -2. -3. -3. -3. -3. -3. -3. -2. -2.]
 [ 0. -1. -2. -2. -2. -3. -3. -3.  0.  0.  0.]
 [ 0. -1. -1. -1. -2. -2. -3.  0.  0.  0.  0.]
 [ 0.  0.  0. -1. -1. -2.  0.  0.  0.  0.  0.]
 [ 1.  1.  0.  0. -1.  0.  0.  1.  1.  1.  0.]
 [ 2.  1.  1.  0.  0. -3. -3.  2.  2.  1.  0.]
 [ 2.  2.  1.  0.  0. -3. -3.  3.  2.  1.  0.]
 [ 3.  2.  1.  0.  0. -2.  0.  3.  2.  0.  0.]
 [ 3.  2.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 3.  2.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
 """
