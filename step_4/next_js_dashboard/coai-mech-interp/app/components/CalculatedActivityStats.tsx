interface CalculatedActivityStatsProps {
    barValues: number[];
    barHeights: number[];
}

export default function CalculatedActivityStats({ barValues, barHeights }: CalculatedActivityStatsProps) {
    const calculateStatistics = (values: number[], heights: number[]) => {
        if (!heights || heights.length === 0) {
            return { mean: 0, stdDev: 0, kurt: 0, skewed: 0 };
        }

        // Normalize weights
        const total = heights.reduce((acc, val) => acc + val, 0);
        const normalizedHeights = heights.map(value => value / total);

        // Weighted mean
        const mean = values.reduce((acc, value, index) => acc + value * normalizedHeights[index], 0);

        // Weighted standard deviation
        const stdDev = Math.sqrt(values.reduce((acc, value, index) => acc + normalizedHeights[index] * Math.pow(value - mean, 2), 0));

        // Weighted kurtosis
        const kurt = values.reduce((acc, value, index) => acc + normalizedHeights[index] * Math.pow((value - mean) / stdDev, 4), 0) - 3;

        // Weighted skewness
        const skewed = values.reduce((acc, value, index) => acc + normalizedHeights[index] * Math.pow((value - mean) / stdDev, 3), 0);

        return { mean, stdDev, kurt, skewed };
    };

    const { mean, stdDev, kurt, skewed } = calculateStatistics(barValues, barHeights);

    return (
        <div className="bg-white p-4"> {/* Set a fixed height */}
            <h4 className="text-[10px] font-semibold mb-2 text-black">Calculated Activity Stats</h4> {/* Set font size to 10px */}
            <table className="w-full text-black text-[10px]"> {/* Set font size to 10px */}
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Standard Deviation</td>
                        <td>{stdDev.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Kurtosis</td>
                        <td>{kurt.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Skewness</td>
                        <td>{skewed.toFixed(2)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    );
}
