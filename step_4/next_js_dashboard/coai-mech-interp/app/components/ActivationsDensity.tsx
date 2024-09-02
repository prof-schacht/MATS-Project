"use client"; // Add this line to mark the component as a client component

import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from 'recharts';

interface ActivationsDensityProps {
  barValues: any;
  barHeights: any;
}

export default function ActivationsDensity({ barValues, barHeights }: ActivationsDensityProps) {
  const data = barValues.map((value: any, index: number) => ({
    value,
    height: barHeights[index],
  }));

  return (
    <div className="bg-white">
      <h4 className="text-[10px] font-semibold mb-2 text-black">Activations Density</h4>
      <ResponsiveContainer width="100%" height={100}>
        <AreaChart data={data} >
          <XAxis dataKey="value" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Area type="monotone" dataKey="height" stroke="#8884d8" fill="#8884d8" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}