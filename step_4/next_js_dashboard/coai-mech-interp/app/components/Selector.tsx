"use client"; // Add this line to mark the component as a client component

import { useState } from 'react';

interface SelectorProps {
  setIndex: (index: number) => void;
  fetchData: (index: number) => void;
}

export default function Selector({ setIndex, fetchData }: SelectorProps) {
  const [index, setLocalIndex] = useState(0);

  const updateData = (newIndex: number) => {
    setLocalIndex(newIndex);
    setIndex(newIndex);
    fetchData(newIndex);
  };

  const handlePrev = () => updateData(Math.max(index - 1, 0));
  const handleNext = () => updateData(index + 1);
  const handleGo = () => updateData(index);

  return (
    <div className="p-4 bg-gray-100 w-full flex items-center justify-center space-x-4">
      <button onClick={handlePrev} className="p-2 bg-gray-300 rounded text-xs">Prev</button>
      <button onClick={handleNext} className="p-2 bg-gray-300 rounded text-xs">Next</button>
      <select className="p-2 border border-gray-300 rounded text-black text-xs">
        <option value="model1">PHI3-mini-instruct</option>
      </select>
      <select className="p-2 border border-gray-300 rounded text-black text-xs">
        <option value="sae1">1B-dclm</option>
      </select>
      <select className="p-2 border border-gray-300 rounded text-black text-xs">
        <option value="layer1">16</option>
      </select>
      <input
        type="number"
        value={index}
        onChange={(e) => setLocalIndex(Number(e.target.value))}
        onClick={(e) => e.target.value = ''}
        className="p-2 border border-gray-300 rounded w-20 text-center text-black text-xs"
        placeholder="Enter index" // Add placeholder attribute
      />
      <button onClick={handleGo} className="p-2 bg-blue-500 text-white rounded text-xs">Go</button>
    </div>
  );
}