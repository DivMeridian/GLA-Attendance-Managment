import React, { useState } from "react";
import axios from "axios";

function DetectAndRecognize() {
  const [file, setFile] = useState(null);
  const [section, setSection] = useState("");
  const [resultImage, setResultImage] = useState(null);
  const [identifiedNames, setIdentifiedNames] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("section", section);

    try {
      const response = await axios.post("http://localhost:8000/detect_and_recognize/", formData);
      setResultImage(`data:image/jpeg;base64,${response.data.image_base64}`);
      setIdentifiedNames(response.data.identified_names || []);
    } catch (error) {
      console.error("Error detecting faces:", error);
    }
  };

  return (
    <div className="max-w-lg mx-auto bg-white p-6 rounded-md shadow-md">
      <h2 className="text-xl font-bold mb-4">Detect and Recognize</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium">Image</label>
          <input
            type="file"
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium">Section</label>
          <input
            type="text"
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            value={section}
            onChange={(e) => setSection(e.target.value)}
            required
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
        >
          Detect
        </button>
      </form>
      {resultImage && (
        <div className="mt-6">
          <h3 className="text-lg font-bold mb-2">Processed Image</h3>
          <img src={resultImage} alt="Processed" className="w-full rounded-md" />
        </div>
      )}
      {identifiedNames.length > 0 && (
        <div className="mt-4">
          <h3 className="text-lg font-bold">Identified Names:</h3>
          <ul className="list-disc list-inside">
            {identifiedNames.map((name, idx) => (
              <li key={idx}>{name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default DetectAndRecognize;
