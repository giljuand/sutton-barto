import numpy as np
import random
import matplotlib.pyplot as plt
import math

def incremental_epsilon_greedy(epsilon):
    avg_reward = 0 #reward during final 100,000 steps
    test_bed = [0]*10 #values of q_*
    Q_values = [0]*10
    lever_count = [0]*10

    for i in range(200000):
        optimal = test_bed.index(max(test_bed)) #identifies optimal lever

        if random.random()<epsilon: #exploration
            choice = random.randint(0,9)
        else: #chooses lever with best value estimate Q
            choice = Q_values.index(max(Q_values))
        
        lever_count[choice]+=1
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: avg_reward += reward/100000

        #changes estimated value Q of lever
        step_size = 1/lever_count[choice]
        Q_values[choice] = Q_values[choice]+step_size*(reward-Q_values[choice])

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return avg_reward

def constant_epsilon_greedy(epsilon, alpha=0.1):
    avg_reward = 0
    test_bed = [0]*10 #values of q_*
    Q_values = [0]*10

    for i in range(200000):
        optimal = test_bed.index(max(test_bed)) #identifies optimal lever

        if random.random()<epsilon: #exploration
            choice = random.randint(0,9)
        else: #chooses lever with best value estimate Q
            choice = Q_values.index(max(Q_values))
        
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: avg_reward += reward/100000

        #changes estimated value Q of lever
        step_size = alpha
        Q_values[choice] = Q_values[choice]+step_size*(reward-Q_values[choice])

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return avg_reward

def incremental_UCB(c):
    avg_reward = 0
    test_bed = [0]*10 #values of q_*
    Q_values = [0]*10
    lever_count = [0]*10

    for i in range(200000):

        preferences = []
        for j in range(10):
            if lever_count[j]==0:
                preferences.append(float("inf"))
            else:
                preferences.append(Q_values[j] + c*(math.log(i+1)/lever_count[j])**0.5)
        choice = preferences.index(max(preferences))
        
        lever_count[choice]+=1
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: avg_reward += reward/100000

        #changes estimated value Q of lever
        step_size = 1/lever_count[choice]
        Q_values[choice] = Q_values[choice]+step_size*(reward-Q_values[choice])

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return avg_reward

def constant_UCB(c, alpha=0.1):
    test_bed = [0]*10 #values of q_*
    Q_values = [0]*10
    lever_count = [0]*10
    avg_reward = 0

    for i in range(200000):

        preferences = []
        for j in range(10):
            if lever_count[j]==0:
                preferences.append(float("inf"))
            else:
                preferences.append(Q_values[j] + c*(math.log(i+1)/lever_count[j])**0.5)
        choice = preferences.index(max(preferences))
        
        lever_count[choice]+=1
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: avg_reward += reward/100000

        #changes estimated value Q of lever
        step_size = alpha
        Q_values[choice] = Q_values[choice]+step_size*(reward-Q_values[choice])

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return avg_reward

def incremental_gradient(alpha):
    result = 0
    avg_reward = 0
    test_bed = [0]*10 #values of q_*
    H_values = [0]*10

    for i in range(200000):

        preferences = []
        denominator = sum([math.exp(H) for H in H_values])
        for j in range(10):
            preferences.append(math.exp(H_values[j])/denominator)

        rand_num = random.random()
        for j in range(10):
            if preferences[j]>rand_num:
                choice = j
                break
            else:
                rand_num-=preferences[j]
        
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: result += reward/100000

        avg_reward = avg_reward + 1/(i+1)*(reward-avg_reward)

        #changes preferences H of levers
        for j in range(10):
            if j == choice:
                H_values[j]=H_values[j]+alpha*(reward-avg_reward)*(1-preferences[j])
            else:
                H_values[j]=H_values[j]-alpha*(reward-avg_reward)*preferences[j]

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return result

def constant_gradient(alpha):
    result = 0
    avg_reward = 0
    test_bed = [0]*10 #values of q_*
    H_values = [0]*10

    for i in range(200000):

        preferences = []
        denominator = sum([math.exp(H) for H in H_values])
        for j in range(10):
            preferences.append(math.exp(H_values[j])/denominator)

        rand_num = random.random()
        for j in range(10):
            if preferences[j]>rand_num:
                choice = j
                break
            else:
                rand_num-=preferences[j]
        
        reward = np.random.normal() + test_bed[choice]
        if i>=100000: result += reward/100000

        avg_reward = avg_reward + 0.1*(reward-avg_reward)

        #changes preferences H of levers
        for j in range(10):
            if j == choice:
                H_values[j]=H_values[j]+alpha*(reward-avg_reward)*(1-preferences[j])
            else:
                H_values[j]=H_values[j]-alpha*(reward-avg_reward)*preferences[j]

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return result

TRIALS=10
inc_ep_greedy = [0]*10
const_ep_greedy = [0]*10
inc_UCB = [0]*10
const_UCB = [0]*10
inc_grad = [0]*10
const_grad = [0]*10
x_axis = [2**(i-7) for i in range(10)]
print(x_axis)
for j in range(TRIALS): #20 trials
    print(j)
    for i in range(10):
        value = 2**(i-7)
        inc_ep_greedy[i]+=incremental_epsilon_greedy(value)/TRIALS
        const_ep_greedy[i]+=constant_epsilon_greedy(value)/TRIALS
        inc_UCB[i]+=incremental_UCB(value)/TRIALS
        const_UCB[i]+=constant_UCB(value)/TRIALS
        inc_grad[i]+=incremental_gradient(value)/TRIALS
        const_grad[i]+=constant_gradient(value)/TRIALS
plt.xscale('log')
plt.plot(x_axis, inc_ep_greedy, 'g-')
plt.plot(x_axis, const_ep_greedy, 'b-')
plt.plot(x_axis, inc_UCB, 'r-')
plt.plot(x_axis, const_UCB, 'y-')
plt.plot(x_axis, inc_grad, 'p-')
plt.plot(x_axis, const_grad, 'o-')
plt.show()
