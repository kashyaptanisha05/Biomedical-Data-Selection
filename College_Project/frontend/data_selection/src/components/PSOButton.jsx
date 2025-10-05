import React, { useState } from "react";

function PSOButton() {
  const [result, setResult] = useState(null);

  const runPSO = async () => {
    const res = await fetch("http://127.0.0.1:5000/run_pso");
    const data = await res.json();
    setResult(data);
  };

  return (
    <div>
      <h2>Run Particle Swarm Optimization</h2>
      <button onClick={runPSO}>Run PSO</button>
      {result && (
        <p>
          Best Position: {result.best_position.join(", ")} <br />
          Accuracy: {result.accuracy}
        </p>
      )}
    </div>
  );
}

export default PSOButton;
