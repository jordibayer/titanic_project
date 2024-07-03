import React from "react";

const Spinner = () => {
  return (
    <div className="flex items-center justify-center my-12">
      <div className="animate-spin rounded-full h-8 w-8 border-t-4 border-b-4 border-blue-500"></div>
    </div>
  );
};

export default Spinner;