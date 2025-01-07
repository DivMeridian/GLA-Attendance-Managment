import React, { useState } from "react";
import axios from "axios";

function RegisterPerson() {
  const [file, setFile] = useState(null);
  const [label, setLabel] = useState("");
  const [contact, setContact] = useState("");
  const [section, setSection] = useState("");
  const [message, setMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const validateInputs = () => {
    const phoneRegex = /^[6-9]\d{9}$/; // Validates 10-digit numbers starting with 6-9

    if (!file) return "Please upload an image.";
    if (!label.trim()) return "Name is required.";
    if (!contact.trim() || !phoneRegex.test(contact)) {
      return "Please enter a valid 10-digit mobile number starting with 6, 7, 8, or 9.";
    }
    if (!section.trim()) return "Section is required.";
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setErrorMessage("");
    setLoading(true);

    // Validate inputs
    const validationError = validateInputs();
    if (validationError) {
      setErrorMessage(validationError);
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("label", label);
    formData.append("Contact", contact);
    formData.append("section", section);

    try {
      const response = await axios.post("http://localhost:8000/register_person/", formData);
      setMessage(response.data.message || "Person registered successfully!");
    } catch (error) {
      if (error.response) {
        setErrorMessage(error.response.data.message || "Failed to register person.");
      } else if (error.request) {
        setErrorMessage("No response from server. Please try again later.");
      } else {
        setErrorMessage("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto bg-white p-6 rounded-md shadow-md">
      <h2 className="text-xl font-bold mb-4">Register Person</h2>
      {errorMessage && <p className="text-red-600 mb-4">{errorMessage}</p>}
      {message && <p className="text-green-600 mb-4">{message}</p>}
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
            type="text"
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
          disabled={loading}
        >
          {loading ? "Registering..." : "Register"}
        </button>
      </form>
    </div>
  );
}

export default RegisterPerson;
