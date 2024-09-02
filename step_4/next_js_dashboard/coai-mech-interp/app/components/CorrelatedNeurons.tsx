export default function CorrelatedNeurons({ neuronIndices, neuronPearson, featureL1 }: CorrelatedNeuronsProps) {
  const correlatedData = neuronIndices.map((index, i) => ({
    index: index,
    pCorr: neuronPearson[i],
    cosSim: featureL1[i],
  }));

  return (
    <div className="bg-white p-4"> {/* Set a fixed height */}
      <h4 className="text-[10px] font-semibold mb-2 text-black">Correlated Neurons</h4> {/* Set font size to 10px */}
      <table className="w-full text-black text-[10px]"> {/* Set font size to 10px */}
        <thead>
          <tr>
            <th>Index</th>
            <th>P. Corr</th>
            <th>Cos Sim</th>
          </tr>
        </thead>
        <tbody>
          {correlatedData.map((item) => (
            <tr key={item.index}>
              <td>{item.index}</td>
              <td>{item.pCorr.toFixed(2)}</td>
              <td>{item.cosSim.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}