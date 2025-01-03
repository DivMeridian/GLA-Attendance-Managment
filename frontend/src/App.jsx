import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./Components/Navbar";
import RegisterPerson from "./Components/RegisterPerson";
import DetectAndRecognize from "./Components/DetectAndRecognize";

function App() {
  return (
    <Router>
      <Navbar />
      <div className="p-4">
        <Routes>
          <Route path="/register" element={<RegisterPerson />} />
          <Route path="/detect" element={<DetectAndRecognize />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
