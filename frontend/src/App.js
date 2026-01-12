import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const checkSpam = async () => {
    if (!message.trim()) return alert("Enter a message");

    setLoading(true);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="container">
      <div className="card">
        <h2>SMS Spam Classifier</h2>

        <textarea
          placeholder="Enter SMS message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />

        <button onClick={checkSpam}>
          {loading ? "Checking..." : "Check"}
        </button>

        {result && (
          <div className="result">
            <p>
              Prediction:{" "}
              <span
                className={
                  result.prediction === "Spam" ? "spam" : "ham"
                }
              >
                {result.prediction}
              </span>
            </p>
            <p>Confidence: {result["confidence score"]}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
