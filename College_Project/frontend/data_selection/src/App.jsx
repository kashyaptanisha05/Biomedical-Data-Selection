  // import React from "react";
  // import PredictForm from "./components/PredictForm";
  // import GAButton from "./components/GAButton";
  // import PSOButton from "./components/PSOButton";

  // function App() {
  //   return (
  //     <div style={{ padding: "20px" }}>
  //       <h1>Heart Disease Prediction</h1>
  //       <PredictForm />
  //       <hr />
  //       <GAButton />
  //       <PSOButton />
  //     </div>
  //   );
  // }

  // export default App;


import React, { useState } from "react";

function PredictForm({ numFeatures }) {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    const input = features.split(",").map(Number);
    if (input.length !== numFeatures) {
      setError(`Please enter exactly ${numFeatures} features.`);
      return;
    }
    setError("");
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: input }),
    });
    const data = await res.json();
    setPrediction(data.prediction);
  };

  return (
    <div>
      <h2>Predict Heart Disease</h2>
      <input
        type="text"
        placeholder="Comma-separated features"
        value={features}
        onChange={(e) => setFeatures(e.target.value)}
        style={{ width: "400px" }}
      />
      <button onClick={handleSubmit}>Predict</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

function GAButton() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runGA = async () => {
    setLoading(true);
    const res = await fetch("http://127.0.0.1:5000/run_ga");
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div>
      <h2>Run Genetic Algorithm</h2>
      <button onClick={runGA} disabled={loading}>
        {loading ? "Running..." : "Run GA"}
      </button>
      {result && (
        <div>
          <p>Best Individual: {result.best_individual.join(", ")}</p>
          <p>Accuracy: {result.accuracy}</p>
          <p>Precision: {result.precision}</p>
          <p>Recall: {result.recall}</p>
          <p>F1 Score: {result.f1_score}</p>
        </div>
      )}
    </div>
  );
}

function PSOButton() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runPSO = async () => {
    setLoading(true);
    const res = await fetch("http://127.0.0.1:5000/run_pso");
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div>
      <h2>Run PSO</h2>
      <button onClick={runPSO} disabled={loading}>
        {loading ? "Running..." : "Run PSO"}
      </button>
      {result && (
        <div>
          <p>Best Position: {result.best_position.join(", ")}</p>
          <p>Accuracy: {result.accuracy}</p>
          <p>Precision: {result.precision}</p>
          <p>Recall: {result.recall}</p>
          <p>F1 Score: {result.f1_score}</p>
        </div>
      )}
    </div>
  );
}

function CombinedButton() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runCombined = async () => {
    setLoading(true);
    const res = await fetch("http://127.0.0.1:5000/run_combined");
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div>
      <h2>Run Combined GA + PSO</h2>
      <button onClick={runCombined} disabled={loading}>
        {loading ? "Running..." : "Run Combined"}
      </button>
      {result && (
        <div>
          <h4>Union Mask</h4>
          <p>Accuracy: {result.union_accuracy}</p>
          <p>Precision: {result.union_precision}</p>
          <p>Recall: {result.union_recall}</p>
          <p>F1 Score: {result.union_f1}</p>
          <h4>Intersection Mask</h4>
          <p>Accuracy: {result.intersection_accuracy}</p>
          <p>Precision: {result.intersection_precision}</p>
          <p>Recall: {result.intersection_recall}</p>
          <p>F1 Score: {result.intersection_f1}</p>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const numFeatures = 13; // Update this to match your dataset columns
  return (
    <div style={{ padding: "20px" }}>
      <h1>Heart Disease Prediction</h1>
      <PredictForm numFeatures={numFeatures} />
      <hr />
      <GAButton />
      <PSOButton />
      <CombinedButton />
    </div>
  );
}
