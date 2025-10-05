import React, { useState } from "react";

function GAButton() {
  const [result, setResult] = useState(null);

  const runGA = async () => {
    const res = await fetch("http://127.0.0.1:5000/run_ga");
    const data = await res.json();
    setResult(data);
  };

  return (
    <div>
      <h2>Run Genetic Algorithm</h2>
      <button onClick={runGA}>Run GA</button>
      {result && (
        <p>
          Best Individual: {result.best_individual} <br />
          Accuracy: {result.accuracy}
        </p>
      )}
    </div>
  );
}

export default GAButton;
