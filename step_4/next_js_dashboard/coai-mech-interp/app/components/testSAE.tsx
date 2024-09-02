import React, { useState } from 'react';

interface TestSAEProps {
  currentFeature: number;
}

const TestSAE: React.FC<TestSAEProps> = ({ currentFeature }) => {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState<{ token: string; activation: number }[]>([]);

  const handleTestActivation = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/feature_test_prompt?prompt=${encodeURIComponent(prompt)}&feature=${currentFeature}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error fetching test activation:', error);
    }
  };

  const getBackgroundColor = (value: number) => {
    if (value === 0) return 'white';
    const greenValue = Math.min(255, Math.floor(value * 255));
    return `rgb(0, ${greenValue}, 0)`;
  };

  return (
    <div className="p-4 border border-gray-300 rounded flex flex-col mt-4">
      <div className="flex items-center">
        <div className="flex-grow">
          <textarea
            className="w-full p-2 border border-gray-300 rounded text-black text-xs"
            rows={1}
            placeholder="Enter your text here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          ></textarea>
        </div>
        <button
          className="ml-2 p-2 bg-blue-500 text-white rounded text-xs"
          onClick={handleTestActivation}
        >
          Test Activation
        </button>
      </div>
      {result.length > 0 && (
        <div className="mt-2 p-2 border border-gray-300 rounded">
          {result.map((item, index) => (
            <span
              key={index}
              style={{
                backgroundColor: getBackgroundColor(item.activation),
                fontSize: '12px',
                display: 'inline',
                margin: 2,
                padding: 0,
                whiteSpace: 'pre-wrap',
                color: 'black', // Set text color to black
              }}
            >
              {item.token}
            </span>
          ))}
        </div>
      )}
      )}
    </div>
  );
};

export default TestSAE;
