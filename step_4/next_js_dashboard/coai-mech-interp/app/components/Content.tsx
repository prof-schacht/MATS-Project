import Tables from './Tables'; // Import Tables component
import Plots from './Plots'; // Import Plots component
import HighlightedTable from './HighlightedTable'; // Import HighlightedTable component
import TestSAE from './testSAE';

interface ContentProps {
  data: any;
  currentFeature: number;
}

export default function Content({ data, currentFeature }: ContentProps) {
  return (
    <div className="flex w-full p-4">
      <div className="w-1/3 p-2">
        <Tables /> {/* Add Tables component in the left 1/3 area */}
      </div>
      <div className="w-2/3 p-2">
        <Plots data={data} /> {/* Pass data to Plots component */}
        <TestSAE currentFeature={currentFeature} />
        <HighlightedTable data={data} /> {/* Add HighlightedTable component below Plots */}
      </div>
    </div>
  );
}