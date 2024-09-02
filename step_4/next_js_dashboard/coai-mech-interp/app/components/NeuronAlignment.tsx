export default function NeuronAlignment({ indices, values, l1 }) {
  const alignmentData = indices.map((index, i) => ({
    index: index,
    value: values[i],
    percentL1: l1[i],
  }));

  return (
    <div className="bg-white p-4"> {/* Set a fixed height */}
      <h4 className="text-[10px] font-semibold mb-2 text-black">Neuron Alignment</h4> {/* Set font size to 10px */}
      <table className="w-full text-black text-[10px]"> {/* Set font size to 10px */}
        <thead>
          <tr>
            <th>Index</th>
            <th>Value</th>
            <th>% of L1</th>
          </tr>
        </thead>
        <tbody>
          {alignmentData.map((item) => (
            <tr key={item.index}>
              <td>{item.index}</td>
              <td>{item.value.toFixed(2)}</td>
              <td>{item.percentL1.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}