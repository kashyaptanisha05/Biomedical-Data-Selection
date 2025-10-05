import React, { useState } from "react";

function PredictForm() {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async () => {
    const input = features.split(",").map(Number);
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
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default PredictForm;
