export default function LogitsTable({ negStr, negValues, posStr, posValues }) {
  const negativeLogits = negStr.map((word, index) => ({
    word: word,
    value: negValues[index],
  }));

  const positiveLogits = posStr.map((word, index) => ({
    word: word,
    value: posValues[index],
  }));

  return (
    <div className="bg-white p-4"> {/* Removed fixed height for dynamic adjustment */}
      <div className="flex justify-between text-[10px]"> {/* Set font size to 10px */}
        <div>
          <h3 className="font-semibold text-black">Negative Logits</h3>
          <table>
            <tbody>
              {negativeLogits.map((item) => (
                <tr key={item.word}>
                  <td className="pr-4 text-black">{item.word}</td>
                  <td className="text-black">{item.value.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="font-semibold text-black">Positive Logits</h3>
          <table>
            <tbody>
              {positiveLogits.map((item) => (
                <tr key={item.word}>
                  <td className="pr-4 text-black">{item.word}</td>
                  <td className="text-black">{item.value.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}