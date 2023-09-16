import matplotlib.pyplot as plt
import numpy as np
HEAD_PROB = 0.4


def run_episode(capital, state_value):
    actions = []
    for bet in range(0, min(capital, 100 - capital) + 1):
        actions.append(HEAD_PROB * state_value[capital + bet] + (1 - HEAD_PROB) * state_value[capital - bet])
    state_value[capital] = max(actions)
    return state_value


def show_state_value(state_value):
    x = [i for i in range(0, 101)]
    value = np.zeros(101)
    for capital in range(1, 100):
        actions = np.zeros(101)
        for bet in range(1, min(capital, 100 - capital) + 1):
            actions[bet] = (HEAD_PROB * state_value[capital + bet] + (1 - HEAD_PROB) * state_value[capital - bet])
        value[capital] = np.argmax(np.round(actions[1:], 5)) + 1
        # np.round(actions[1:], 5)

    print(f"state_value:{state_value}")
    for i in range(10):
        print(value[i*10:(i+1)*10])
    plt.figure(num=1, figsize=(10, 4))
    plt.plot(x, value)
    plt.savefig('optimal_policy.png')
    plt.close()

    plt.figure(num=0, figsize=(10, 4))
    plt.plot(x, state_value)
    plt.savefig('state_value.png')
    plt.close()

if __name__ == "__main__":
    state_value = np.zeros(101)
    state_value[100] = 1.0
    for epoch in range(100):
        print(f"epoch:{epoch}")
        for capital in range(1, 100):
            state_value = run_episode(capital, state_value)

    show_state_value(state_value)
