                                                                      Feature Selection
import numpy as np
import pandas as pd

# === Objective Function for Feature Selection (maximize classification accuracy) ===
def fitness_function(x, data, labels):
    # Binary selection mask (1 = selected, 0 = not selected)
    selected_features = x > 0.5
    if np.sum(selected_features) == 0:
        return -np.inf  # avoid zero-feature case

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=10)
    try:
        scores = cross_val_score(clf, data[:, selected_features], labels, cv=3, scoring='accuracy')
        return scores.mean()
    except:
        return -np.inf

# === Initialization ===
def initialize_population(n_agents, dim, bounds):
    lower, upper = bounds
    return lower + (upper - lower) * np.random.rand(n_agents, dim)

# === Mean position of population ===
def get_mean_position(pop):
    return np.mean(pop, axis=0)

# === Adaptive param update for AFO ===
def update_adaptive_params(v, delta, fitness_i, fitness_best, t, max_iter):
    v *= np.exp(-abs(fitness_i - fitness_best) / (abs(fitness_best) + 1e-10))
    delta *= (1 - t / max_iter)
    return v, delta

# === Logistic transition control for λ(t) ===
def lambda_schedule(t, t0, c):
    return 1 / (1 + np.exp(-c * (t - t0)))

# === ADMFO Algorithm for Feature Selection ===
def ADMFO_feature_selection(data, labels, n_agents=30, max_iter=50):
    dim = data.shape[1]
    bounds = (0, 1)  # real-coded [0, 1] for binary thresholding

    pop = initialize_population(n_agents, dim, bounds)
    fitness = np.array([fitness_function(ind, data, labels) for ind in pop])
    best_idx = np.argmax(fitness)
    gbest = pop[best_idx].copy()
    gbest_fitness = fitness[best_idx]

    gamma1, gamma2 = 0.5, 0.3
    v, delta = 0.9, 0.5
    t0, c = max_iter // 2, 0.1

    for t in range(max_iter):
        lambda_t = lambda_schedule(t, t0, c)
        mean_pos = get_mean_position(pop)

        for i in range(n_agents):
            R1, R2 = np.random.rand(), np.random.rand()
            r = np.random.uniform(-1, 1, dim)

            if lambda_t < 0.5:
                # === IDMO Phase: Exploration ===
                pop[i] += (1 - lambda_t) * (gamma1 * R1 * (gbest - pop[i])) + \
                          lambda_t * (gamma2 * R2 * (mean_pos - pop[i]))
            else:
                # === AFO Phase: Exploitation ===
                lbest = pop[np.random.randint(n_agents)]
                pop[i] += v * (lbest - pop[i]) + delta * r
                v, delta = update_adaptive_params(v, delta, fitness[i], gbest_fitness, t, max_iter)

            pop[i] = np.clip(pop[i], 0, 1)

        fitness = np.array([fitness_function(ind, data, labels) for ind in pop])
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > gbest_fitness:
            gbest = pop[best_idx].copy()
            gbest_fitness = fitness[best_idx]

        print(f"Iteration {t + 1}/{max_iter} | Best Accuracy: {gbest_fitness:.5f}")

    selected_mask = gbest > 0.5
    return selected_mask, gbest_fitness

# === Main Execution ===
if __name__ == "__main__":
    # === Load CSV Data ===
    input_csv = "/content/drive/MyDrive/Colab Notebooks/archive (75)/feature_output_labeled.csv"
    output_csv = "/content/drive/MyDrive/Colab Notebooks/archive (75)/selected_features_output.csv"
    label_column = 'label'  # replace with actual label column name

    df = pd.read_csv(input_csv)
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    # === Run ADMFO for Feature Selection ===
    selected_mask, best_score = ADMFO_feature_selection(X, y, n_agents=30, max_iter=50)

    # === Save selected features ===
    selected_columns = df.drop(columns=[label_column]).columns[selected_mask]
    selected_df = df[selected_columns]
    selected_df[label_column] = y
    selected_df.to_csv(output_csv, index=False)

    print(f"\n✅ Selected features saved to: {output_csv}")
    print(f"✔️ Selected Features: {list(selected_columns)}")