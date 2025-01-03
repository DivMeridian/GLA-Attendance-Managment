import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="bg-blue-500 text-white p-4 shadow-md">
      <div className="flex justify-between items-center">
        <h1 className="text-lg font-bold">Face Recognition</h1>
        <div className="space-x-4">
          <Link to="/register" className="hover:underline">
            Register
          </Link>
          <Link to="/detect" className="hover:underline">
            Detect
          </Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
