interface HighlightedTableProps {
  data: any;
}

const HighlightedTable = ({ data }: HighlightedTableProps) => {
  const { activations } = data;

  const getBackgroundColor = (value: number) => {
    if (value === 0) return 'white';
    const greenValue = Math.min(255, Math.floor(value * 255));
    return `rgb(0, ${greenValue}, 0)`;
  };

  const processToken = (token: string) => {
    return token.replace(/▁/g, ' ');
  };

  return (
    <div className="mt-4 bg-white text-black p-2 w-full border border-gray-300 rounded-md">
      {activations.map((activation: any, index: number) => (
        <div key={index} className="mb-2 border-b border-gray-300">
          {activation.tokens.map((token: string, tokenIndex: number) => (
            <span
              key={tokenIndex}
              style={{
                backgroundColor: getBackgroundColor(activation.values[tokenIndex]),
                fontSize: '10px',
                display: 'inline',
                margin: 0,
                padding: 0,
                whiteSpace: 'pre-wrap', // Preserve spaces and handle line breaks
              }}
            >
              {processToken(token)}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
};

export default HighlightedTable;