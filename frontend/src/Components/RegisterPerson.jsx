import React, { useState } from "react";
import axios from "axios";

function RegisterPerson() {
  const [file, setFile] = useState(null);
  const [label, setLabel] = useState("");
  const [contact, setContact] = useState("");
  const [section, setSection] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("label", label);
    formData.append("Contact", contact);
    formData.append("section", section);

    try {
      const response = await axios.post("http://localhost:8000/register_person/", formData);
      setMessage(response.data.message);
    } catch (error) {
      console.error("Error registering person:", error);
      setMessage("Failed to register person.");
    }
  };

  return (
    <div className="max-w-lg mx-auto bg-white p-6 rounded-md shadow-md">
      <h2 className="text-xl font-bold mb-4">Register Person</h2>
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
          <label className="block text-sm font-medium">Name</label>
          <input
            type="text"
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium">Contact</label>
          <input
            type="number"
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            value={contact}
            onChange={(e) => setContact(e.target.value)}
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
          Register
        </button>
      </form>
      {message && <p className="mt-4 text-green-600">{message}</p>}
    </div>
  );
}

export default RegisterPerson;
