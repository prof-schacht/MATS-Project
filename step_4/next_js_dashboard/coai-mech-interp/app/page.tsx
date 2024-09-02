"use client"; // Add this line to mark the component as a client component

import { useState, useEffect } from 'react';
import Image from "next/image";
import Navigation from './components/Navigation'; // Import Navigation component
import Footer from './components/Footer'; // Import Footer component
import Selector from './components/Selector'; // Import Selector component
import Content from './components/Content'; // Import Content component

export default function Home() {
  const [data, setData] = useState<any>(null);
  const [index, setIndex] = useState<number>(0);
  const [error, setError] = useState<string | null>(null); // Add state for error handling

  const fetchData = async (index: number) => {
    try {
      console.log(`Fetching data for index: ${index}`); // Add logging
      const response = await fetch(`http://127.0.0.1:8000/feature/${index}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setData(result);
      setError(null); // Clear any previous errors
    } catch (err: unknown) {
      console.error(err); // Log the error
      if (err instanceof Error) {
        setError(err.message); // Set error message
      } else {
        setError('An unknown error occurred');
      }
    }
  };

  useEffect(() => {
    fetchData(index);
  }, [index]);

  return (
    <div className="flex flex-col min-h-screen bg-white"> {/* Set up a flex column layout with minimum height */}
      <Navigation /> {/* Add Navigation component at the top */}
      <Selector setIndex={setIndex} fetchData={fetchData} /> {/* Pass setIndex and fetchData to Selector */}
      {error && <div className="text-red-500">{error}</div>} {/* Display error message if any */}
      {data && <Content data={data} />} {/* Pass data to Content component */}
      <div className="mt-auto w-full"> {/* Wrapper div to ensure Footer sticks to the bottom */}
        <Footer /> {/* Add Footer component at the bottom */}
      </div>
    </div>
  );
}
