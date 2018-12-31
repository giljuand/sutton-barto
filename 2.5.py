import numpy as np
import random
import matplotlib.pyplot as plt

def nonstationary_bandit(steps, epsilon, incremental=True, alpha = 0.1):
    reward_log = []
    optimal_action_log = []
    test_bed = [0]*10 #values of q_*
    Q_values = [0]*10
    if incremental: lever_count = [0]*10

    for i in range(steps):
        optimal = test_bed.index(max(test_bed)) #identifies optimal lever

        if random.random()<epsilon: #exploration
            choice = random.randint(0,9)
        else: #chooses lever with best value estimate Q
            choice = Q_values.index(max(Q_values))
        
        if incremental:  lever_count[choice]+=1
        reward = np.random.normal() + test_bed[choice]
        reward_log.append(reward) #logs reward
        optimal_action_log.append(optimal==choice) #logs whether optimal lever pulled

        #changes estimated value Q of lever
        if incremental:
            step_size = 1/lever_count[choice]
        else:
            step_size = alpha
        Q_values[choice] = Q_values[choice]+step_size*(reward-Q_values[choice])

        # changes distribution of bandit arms
        for j in range(10):
            test_bed[j]+=np.random.normal(0,0.01)
    return (reward_log, optimal_action_log)


STEPS = 10000
TRIALS = 2000
EPSILON = 0.1
inc_optimal_rates = [0]*STEPS
inc_avg_reward = [0]*STEPS
alpha_optimal_rates = [0]*STEPS
alpha_avg_reward = [0]*STEPS

for i in range(TRIALS):
    print(i)
    #incremental policy
    inc_results = nonstationary_bandit(STEPS, EPSILON, True)
    for i in range(STEPS):
        inc_optimal_rates[i]+=int(inc_results[1][i])/TRIALS
        inc_avg_reward[i]+=inc_results[0][i]/TRIALS
    
    #constant step-size
    alpha_results = nonstationary_bandit(STEPS, EPSILON, False)
    for i in range(STEPS):
        alpha_optimal_rates[i]+=int(alpha_results[1][i])/TRIALS
        alpha_avg_reward[i]+=alpha_results[0][i]/TRIALS

x_axis = list(range(STEPS))
plt.figure(1)
plt.subplot(211)
plt.plot(x_axis, alpha_avg_reward, 'r-')
plt.plot(x_axis, inc_avg_reward, 'b-')

plt.subplot(212)
plt.plot(x_axis, alpha_optimal_rates, 'r-')
plt.plot(x_axis, inc_optimal_rates, 'b-')

plt.show()


