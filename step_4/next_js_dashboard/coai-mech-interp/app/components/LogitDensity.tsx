import React from 'react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from 'recharts';

interface LogitDensityProps {
  barValues: number[];
  barHeights: number[];
}

export default function LogitDensity({ barValues, barHeights }: LogitDensityProps) {
  const data = barValues.map((value, index) => ({
    value,
    height: barHeights[index],
    negativeHeight: value < 0 ? barHeights[index] : 0,
    positiveHeight: value >= 0 ? barHeights[index] : 0,
  }));

  return (
    <div className="bg-white p-4">
      <h4 className="text-sm font-semibold mb-2 text-black">Logit Density</h4>
      <ResponsiveContainer width="100%" height={100}>
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="value"
            tick={{ fontSize: 10 }}
            tickFormatter={(value) => value.toFixed(1)}
            domain={['dataMin', 'dataMax']}
          />
          <YAxis tick={{ fontSize: 10 }} />
          <Area
            type="monotone"
            dataKey="negativeHeight"
            stroke="#ff8080"
            fill="#ff8080"
            fillOpacity={0.8}
          />
          <Area
            type="monotone"
            dataKey="positiveHeight"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.8}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}