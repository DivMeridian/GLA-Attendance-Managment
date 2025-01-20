import React, { useState, useEffect } from "react";
import axios from "axios";

function DetectAndRecognize() {
  const [files, setFiles] = useState([]);
  const [section, setSection] = useState("");
  const [sections, setSections] = useState([]);
  const [resultImages, setResultImages] = useState([]);
  const [identifiedNames, setIdentifiedNames] = useState([]);
  const [registeredUsers, setRegisteredUsers] = useState([]);
  const [attendance, setAttendance] = useState([]);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  // const backendURI = 'https://attendance-backend.azurewebsites.net'
  const backendURI = "http://localhost:8000";

  // Fetch available sections from the backend
  useEffect(() => {
    const fetchSections = async () => {
      try {
        const response = await axios.get(`${backendURI}/get_sections/`);
        setSections(response.data.sections || []);
      } catch (error) {
        console.error("Error fetching sections:", error);
      }
    };
    fetchSections();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage("");
    setLoading(true);
    setResultImages([]);
    setIdentifiedNames([]);
  
    if (files.length === 0 || !section.trim()) {
      setErrorMessage("Please provide all required inputs.");
      setLoading(false);
      return;
    }
  
    const responses = [];
    try {
      // Send each file with its own API call
      for (let file of files) {
        const formData = new FormData();
        formData.append("file", file); // Use "file" instead of "files"
        formData.append("section", section);
  
        const response = await axios.post(`${backendURI}/detect_and_recognize/`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        responses.push(response.data);
      }
  
      // Combine results from all responses
      const combinedImages = responses.flatMap((res) => res.image_base64 || []);
      const combinedNames = responses.flatMap((res) => res.identified_names || []);
      
      setResultImages(combinedImages);
      setIdentifiedNames(combinedNames);
  
      // Fetch registered users
      const usersResponse = await axios.get(`${backendURI}/get_registered_users/${section}`);
      setRegisteredUsers(usersResponse.data.registered_users || []);
    } catch (error) {
      if (error.response) {
        setErrorMessage(error.response.data.message || "Failed to process the request.");
      } else if (error.request) {
        setErrorMessage("No response from server. Please try again later.");
      } else {
        setErrorMessage("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    if (registeredUsers.length > 0) {
      const attendanceList = registeredUsers.map((user) => ({
        name: user,
        present: identifiedNames.includes(user),
      }));
      setAttendance(attendanceList);
    }
  }, [registeredUsers, identifiedNames]);

  const toggleAttendance = (index) => {
    const updatedAttendance = [...attendance];
    updatedAttendance[index].present = !updatedAttendance[index].present;
    setAttendance(updatedAttendance);
  };

  const submitAttendance = async () => {
    setErrorMessage("");
    setLoading(true);

    try {
      await axios.post(`${backendURI}/submit_attendance/`, {
        section,
        attendance,
      });
      setShowConfirmation(true);
    } catch (error) {
      if (error.response) {
        setErrorMessage(error.response.data.message || "Failed to submit attendance.");
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
      <h2 className="text-xl font-bold mb-4">Detect and Recognize</h2>
      {errorMessage && <p className="text-red-600 mb-4">{errorMessage}</p>}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium">Images</label>
          <input
            type="file"
            multiple
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            onChange={(e) => setFiles(e.target.files)}
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium">Section</label>
          <select
            className="mt-1 block w-full border border-gray-300 rounded-md p-2"
            value={section}
            onChange={(e) => setSection(e.target.value)}
            required
          >
            <option value="">-- Select a Section --</option>
            {sections.map((sec) => (
              <option key={sec} value={sec}>
                {sec}
              </option>
            ))}
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? "Processing..." : "Detect"}
        </button>
      </form>
      {resultImages.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-bold mb-2">Processed Images</h3>
          <div className="grid grid-cols-1 gap-4">
            {resultImages.map((image, index) => (
              <img
                key={index}
                src={`data:image/jpeg;base64,${image}`}
                alt={`Processed ${index + 1}`}
                className="w-full rounded-md"
              />
            ))}
          </div>
        </div>
      )}
      {attendance.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-bold mb-2">Attendance Record</h3>
          <table className="table-auto w-full border-collapse border border-gray-300">
            <thead>
              <tr>
                <th className="border border-gray-300 px-4 py-2">Name</th>
                <th className="border border-gray-300 px-4 py-2">Present</th>
              </tr>
            </thead>
            <tbody>
              {attendance.map((record, index) => (
                <tr key={index}>
                  <td className="border border-gray-300 px-4 py-2">{record.name}</td>
                  <td className="border border-gray-300 px-4 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={record.present}
                      onChange={() => toggleAttendance(index)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            className="mt-4 w-full bg-green-500 text-white p-2 rounded-md hover:bg-green-600"
            onClick={submitAttendance}
            disabled={loading}
          >
            {loading ? "Submitting..." : "Submit Attendance"}
          </button>
        </div>
      )}
      {showConfirmation && (
        <div className="mt-6 bg-green-100 p-4 rounded-md text-green-800">
          <h3 className="text-lg font-bold">Attendance Submitted!</h3>
          <p>Attendance has been successfully submitted for section {section}.</p>
        </div>
      )}
    </div>
  );
}

export default DetectAndRecognize;
