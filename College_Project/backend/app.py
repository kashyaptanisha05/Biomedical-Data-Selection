# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ga_pso_svm import run_ga, run_pso, train_final_svm, predict_svm

# app = Flask(__name__)
# CORS(app)

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     features = data.get("features")
#     if not features:
#         return jsonify({"error":"No features provided"}), 400
#     pred = predict_svm(features)
#     prediction_text = "No Heart Disease" if pred == 0 else "Heart Disease"
#     return jsonify({"prediction": prediction_text})

# @app.route("/run_ga", methods=["GET"])
# def ga():
#     best_ind, acc = run_ga()
#     train_final_svm(best_ind)
#     return jsonify({"best_individual": best_ind, "accuracy": acc})

# @app.route("/run_pso", methods=["GET"])
# def pso():
#     mask, acc = run_pso()
#     train_final_svm(mask)
#     return jsonify({"best_position": mask.tolist(), "accuracy": float(acc)})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from ga_pso_svm import (
    load_dataset,
    preprocess,
    split_data,
    run_ga,
    run_pso,
    combine_masks,
    train_final_svm
)
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# ------------------------
# Load and preprocess dataset once at startup
X, y, feature_names = load_dataset()
X_scaled = preprocess(X)
X_train, X_test, y_train, y_test = split_data(X_scaled, y)


# ------------------------
# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features")
    if not features or len(features) != X.shape[1]:
        return jsonify({"error": f"Please provide {X.shape[1]} features"}), 400

    try:
        clf = joblib.load("svm_model.pkl")
        mask = np.load("feature_mask.npy").astype(bool)
        scaler = joblib.load("scaler.pkl")

        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        pred = clf.predict(features_scaled[:, mask])[0]

        return jsonify({
            "prediction": "Heart Disease" if pred == 1 else "No Heart Disease"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------
# Run GA
@app.route("/run_ga", methods=["GET"])
def ga_endpoint():
    best_ind, acc, cm, precision, recall, f1 = run_ga(X_train, X_test, y_train, y_test)
    train_final_svm(best_ind, X_train, X_test, y_train, y_test)

    return jsonify({
        "best_individual": list(best_ind),
        "accuracy": acc,
        "confusion_matrix": cm,  # already a list
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })



# ------------------------
# Run PSO
@app.route("/run_pso", methods=["GET"])
def pso_endpoint():
    mask, acc, cm, precision, recall, f1 = run_pso(X_train, X_test, y_train, y_test)
    train_final_svm(mask, X_train, X_test, y_train, y_test)

    return jsonify({
        "best_position": np.array(mask).astype(int).tolist(),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


# ------------------------
# Combined GA + PSO
@app.route("/run_combined", methods=["GET"])
def combined_endpoint():
    ga_mask, _, _, _, _, _ = run_ga(X_train, X_test, y_train, y_test)
    pso_mask, _, _, _, _, _ = run_pso(X_train, X_test, y_train, y_test)

    combined_union = combine_masks(ga_mask, pso_mask, method="union")
    combined_intersection = combine_masks(ga_mask, pso_mask, method="intersection")
    
    # Train with union
    _, acc_u, cm_u, p_u, r_u, f1_u = train_final_svm(combined_union, X_train, X_test, y_train, y_test)
    
    # Train with intersection
    _, acc_i, cm_i, p_i, r_i, f1_i = train_final_svm(combined_intersection, X_train, X_test, y_train, y_test)
    
    return jsonify({
        "union_mask": combined_union,  # already a list
        "union_accuracy": acc_u,
        "union_precision": p_u,
        "union_recall": r_u,
        "union_f1": f1_u,
        "intersection_mask": combined_intersection,  # already a list
        "intersection_accuracy": acc_i,
        "intersection_precision": p_i,
        "intersection_recall": r_i,
        "intersection_f1": f1_i
    })


# ------------------------
# Main
if __name__ == "__main__":
    app.run(debug=True)
