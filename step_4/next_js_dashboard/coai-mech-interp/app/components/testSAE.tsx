import React from 'react';

const TestSAE = () => {
  return (
    <div className="p-4 border border-gray-300 rounded flex items-center mt-4"> {/* Added margin-top of 2px */}
      <div className="flex-grow">
        <textarea className="w-full p-2 border border-gray-300 rounded text-black text-xs" rows={1} placeholder="Enter your text here..."></textarea>
      </div>
      <button className="ml-2 p-2 bg-blue-500 text-white rounded text-xs">Test Activation</button>
    </div>
  );
};

export default TestSAE;
