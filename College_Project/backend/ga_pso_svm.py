# import numpy as np
# import pandas as pd
# import random
# import joblib
# import zipfile
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from deap import base, creator, tools, algorithms
# import pyswarms as ps

# # ------------------------
# # 1️⃣ Extract Kaggle zip
# zip_path = "heart_disease_dataset.zip"
# extract_folder = "dataset"
# os.makedirs(extract_folder, exist_ok=True)

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_folder)

# # Find CSV automatically
# csv_files = [f for f in os.listdir(extract_folder) if f.endswith(".csv")]
# if not csv_files:
#     raise FileNotFoundError("No CSV found in the zip folder!")
# csv_file = os.path.join(extract_folder, csv_files[0])
# print("Using CSV file:", csv_file)

# # ------------------------
# # 2️⃣ Load dataset
# df = pd.read_csv(csv_file)

# # Assume target column is named 'target' (update if needed)
# if 'target' not in df.columns:
#     raise ValueError("CSV must have a 'target' column for labels")

# X = df.drop("target", axis=1).values
# y = df["target"].values
# num_features = X.shape[1]

# # ------------------------
# # 3️⃣ Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# joblib.dump(scaler, "scaler.pkl")

# # ------------------------
# # 4️⃣ Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.3, random_state=42, stratify=y
# )

# # ------------------------
# # 5️⃣ Helper: evaluate mask
# def evaluate_mask(mask):
#     mask = np.array(mask).astype(bool)
#     if mask.sum() == 0:
#         return 0.0
#     clf = SVC(kernel="rbf", C=1.0, gamma="scale")
#     clf.fit(X_train[:, mask], y_train)
#     y_pred = clf.predict(X_test[:, mask])
#     return accuracy_score(y_test, y_pred)

# # ------------------------
# # 6️⃣ Genetic Algorithm
# def run_ga(NGEN=25, POP=40, CX_PB=0.5, MUT_PB=0.2, INDPB=0.05):
#     if "FitnessMax" in creator.__dict__:
#         del creator.FitnessMax
#     if "Individual" in creator.__dict__:
#         del creator.Individual

#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)

#     toolbox = base.Toolbox()
#     toolbox.register("attr_bool", random.randint, 0, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     toolbox.register("evaluate", lambda ind: (evaluate_mask(ind),))

#     pop = toolbox.population(n=POP)
#     hall_of_fame = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("max", np.max)
#     stats.register("min", np.min)

#     pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=NGEN,
#                                        stats=stats, halloffame=hall_of_fame, verbose=False)
#     best_ind = hall_of_fame[0]
#     best_acc = evaluate_mask(best_ind)
#     return best_ind, best_acc

# # ------------------------
# # 7️⃣ Particle Swarm Optimization
# def run_pso(iters=25, n_particles=40):
#     def pso_cost(x):
#         cost = np.zeros(x.shape[0])
#         for i in range(x.shape[0]):
#             mask = x[i] > 0.5
#             if mask.sum() == 0:
#                 cost[i] = 1.0
#                 continue
#             clf = SVC(kernel="rbf", C=1.0, gamma="scale")
#             clf.fit(X_train[:, mask], y_train)
#             y_pred = clf.predict(X_test[:, mask])
#             cost[i] = 1.0 - accuracy_score(y_test, y_pred)
#         return cost

#     options = {
#     'c1': 0.5,   # cognitive parameter
#     'c2': 0.3,   # social parameter
#     'w': 0.9,    # inertia weight
#     'k': 3,      # number of neighbors in ring topology
#     'p': 2       # Minkowski p-norm, must be >= 1
# }

#     optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=num_features, options=options)
#     cost, pos = optimizer.optimize(pso_cost, iters=iters, verbose=False)
#     mask = np.array(pos).astype(bool)
#     acc = 1 - cost
#     return mask, acc

# # ------------------------
# # 8️⃣ Train final SVM
# def train_final_svm(mask):
#     mask = np.array(mask).astype(bool)
#     clf = SVC(kernel="rbf", C=1.0, gamma="scale")
#     clf.fit(X_train[:, mask], y_train)
#     joblib.dump(clf, "svm_model.pkl")
#     np.save("feature_mask.npy", mask.astype(int))
#     return clf

# # ------------------------
# # 9️⃣ Predict using saved model
# def predict_svm(features):
#     clf = joblib.load("svm_model.pkl")
#     mask = np.load("feature_mask.npy").astype(bool)
#     scaler = joblib.load("scaler.pkl")  # Load the scaler
#     features = np.array(features).reshape(1, -1)
#     features_scaled = scaler.transform(features)  # Scale input
#     features_masked = features_scaled[:, mask]   # Apply mask
#     return int(clf.predict(features_masked)[0])

# # train_model.py
# from ga_pso_svm import run_ga, train_final_svm

# # Run GA to select best features
# best_mask, best_acc = run_ga(NGEN=25, POP=40)
# print("Best feature mask from GA:", best_mask)
# print("Best accuracy:", best_acc)

# # Train final SVM using selected features
# train_final_svm(best_mask)
# print("Final SVM trained and saved as 'svm_model.pkl'")

# ga_pso_svm.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend (safe for Flask threads)
import matplotlib.pyplot as plt
import random
import os
import joblib
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from deap import base, creator, tools, algorithms
import pyswarms as ps

# ------------------------
# Dataset folder (used by your loader if you need it)
DATASET_FOLDER = "dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)


# ------------------------
# 1️⃣ Load / Update Dataset helper (optional)
def load_dataset(folder=DATASET_FOLDER):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found in dataset folder!")
    latest_file = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    if "target" not in df.columns:
        raise ValueError("CSV must have a 'target' column!")
    X = df.drop("target", axis=1).values
    y = df["target"].values
    feature_names = df.drop("target", axis=1).columns.tolist()
    return X, y, feature_names


# ------------------------
# 2️⃣ Preprocessing helper
def preprocess(X, scaler_path="scaler.pkl"):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    return X_scaled


# ------------------------
# 3️⃣ Train-test split helper
def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


# ------------------------
# 4️⃣ Evaluate feature mask (always returns same shaped tuple)
def evaluate_mask(mask, X_train, X_test, y_train, y_test):
    """
    mask: array-like of booleans / 0-1
    returns: (acc, confusion_matrix, precision, recall, f1)
    """
    mask = np.array(mask).astype(bool)

    # If no features selected -> return zero metrics (consistent type)
    if mask.sum() == 0:
        # Try to infer number of classes for confusion matrix shape
        classes = np.unique(y_train)
        n_classes = len(classes)
        cm_zero = np.zeros((n_classes, n_classes), dtype=int)
        return 0.0, cm_zero, 0.0, 0.0, 0.0

    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    clf.fit(X_train[:, mask], y_train)
    y_pred = clf.predict(X_test[:, mask])

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # Use zero_division=0 to avoid warnings if a label is missing
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return acc, cm, precision, recall, f1


# ------------------------
# Utility: safe recreate DEAP creators (avoid errors on repeated calls)
def recreate_deap_creators():
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


# ------------------------
# 5️⃣ Genetic Algorithm (returns mask + metrics)
def run_ga(X_train, X_test, y_train, y_test, NGEN=25, POP=40, CX_PB=0.5, MUT_PB=0.2, INDPB=0.05):
    """
    Run GA for feature selection.
    Returns: (mask (list of 0/1), acc, cm, precision, recall, f1)
    """
    num_features = X_train.shape[1]

    # recreate creators to avoid double-creation error when called multiple times
    recreate_deap_creators()

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # evaluator returns a tuple; DEAP expects a tuple for fitness
    def _evaluate(individual):
        acc, *_ = evaluate_mask(individual, X_train, X_test, y_train, y_test)
        return (acc,)

    toolbox.register("evaluate", _evaluate)

    pop = toolbox.population(n=POP)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=NGEN,
                                       stats=stats, halloffame=hall_of_fame, verbose=False)

    best_ind = hall_of_fame[0]
    # convert DEAP Individual to plain list
    best_mask = np.array(list(best_ind)).astype(int)

    acc, cm, precision, recall, f1 = evaluate_mask(best_mask, X_train, X_test, y_train, y_test)

    # Plot GA evolution (max and avg fitness from logbook)
    try:
        plt.figure()
        gens = list(range(len(logbook)))
        # logbook.select("max") returns a list of tuples (fitness,), so we extract numeric values
        max_vals = [m[0] if isinstance(m, (list, tuple)) else m for m in logbook.select("max")]
        avg_vals = [a[0] if isinstance(a, (list, tuple)) else a for a in logbook.select("avg")]
        plt.plot(gens, max_vals, label="Max Fitness")
        plt.plot(gens, avg_vals, label="Avg Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title("GA Accuracy Evolution")
        plt.legend()
        plt.savefig("ga_evolution.png")
        plt.close()
    except Exception:
        # plotting should not break GA result
        plt.close()

    return best_mask.tolist(), float(acc), cm.tolist(), float(precision), float(recall), float(f1)


# ------------------------
# 6️⃣ Particle Swarm Optimization (returns mask + metrics)
def run_pso(X_train, X_test, y_train, y_test, iters=25, n_particles=40):
    """
    Binary PSO for feature selection.
    Returns: (mask (list of 0/1), acc, cm, precision, recall, f1)
    """

    num_features = X_train.shape[1]   # <-- FIX: define num_features

    best_cost_history = []

    def pso_cost(x):
        """
        x: array-like shape (n_particles, dimensions)
        returns: array of costs (n_particles,)
        """
        cost = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            mask = x[i] > 0.5
            if mask.sum() == 0:
                cost[i] = 1.0
                continue
            clf = SVC(kernel="rbf", C=1.0, gamma="scale")
            clf.fit(X_train[:, mask], y_train)
            y_pred = clf.predict(X_test[:, mask])
            cost[i] = 1.0 - accuracy_score(y_test, y_pred)
        # track best cost this iteration (min)
        best_cost_history.append(cost.min())
        return cost

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=num_features, options=options)
    # optimizer.optimize returns best_cost (float) and best_pos (array)
    best_cost, best_pos = optimizer.optimize(pso_cost, iters=iters, verbose=False)

    # Plot PSO evolution
    try:
        plt.figure()
        plt.plot(best_cost_history, label="Best Cost (1-Accuracy)")
        plt.xlabel("Iteration")
        plt.ylabel("Cost (1 - Accuracy)")
        plt.title("PSO Evolution")
        plt.legend()
        plt.savefig("pso_evolution.png")
        plt.close()
    except Exception:
        plt.close()

    # convert best position to mask
    mask = np.array(best_pos).astype(int)  # 0/1 ints
    acc, cm, precision, recall, f1 = evaluate_mask(mask, X_train, X_test, y_train, y_test)

# return mask as numpy array (not list)
    return mask, float(acc), cm, float(precision), float(recall), float(f1)



# ------------------------
# 7️⃣ Combine Masks helper
def combine_masks(mask1, mask2, method="union"):
    mask1 = np.array(mask1).astype(bool)
    mask2 = np.array(mask2).astype(bool)
    if method == "union":
        combined = np.logical_or(mask1, mask2)
    elif method == "intersection" or method == "intersect":
        combined = np.logical_and(mask1, mask2)
    else:
        raise ValueError("method must be 'union' or 'intersection'")
    return combined.astype(int).tolist()


# ------------------------
# 8️⃣ Train final SVM and persist model & mask
def train_final_svm(mask, X_train, X_test, y_train, y_test, model_path="svm_model.pkl", mask_path="feature_mask.npy"):
    mask = np.array(mask).astype(bool)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    # If no features selected, raise error to avoid training on empty data
    if mask.sum() == 0:
        raise ValueError("Cannot train final SVM: no features selected by mask.")
    clf.fit(X_train[:, mask], y_train)
    joblib.dump(clf, model_path)
    np.save(mask_path, mask.astype(int))
    acc, cm, precision, recall, f1 = evaluate_mask(mask, X_train, X_test, y_train, y_test)
    return clf, float(acc), cm.tolist(), float(precision), float(recall), float(f1)
