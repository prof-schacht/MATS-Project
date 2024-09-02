import NeuronAlignment from './NeuronAlignment';
import CorrelatedNeurons from './CorrelatedNeurons';
import LogitsTable from './LogitsTable';
import ActivationsDensity from './ActivationsDensity';
import LogitDensity from './LogitDensity';
import CalculatedActivityStats from './CalculatedActivityStats';
interface PlotsProps {
  data: any;
}

export default function Plots({ data }: PlotsProps) {
  return (
    <div className="bg-white p-4 border border-gray-300 rounded">
      <h2 className="text-xl font-semibold mb-4 text-black">Plots</h2>
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-1">
          <NeuronAlignment
            indices={data.neuron_alignment_indices}
            values={data.neuron_alignment_values}
            l1={data.neuron_alignment_l1}
          />
          <CorrelatedNeurons
            neuronIndices={data.correlated_neurons_indices}
            neuronL1={data.correlated_neurons_l1}
            neuronPearson={data.correlated_neurons_pearson}
            featureIndices={data.correlated_features_indices}
            featureL1={data.correlated_features_l1}
            featurePearson={data.correlated_features_pearson}
          />
          <CalculatedActivityStats
            barValues={data.logits_hist_data_bar_values} // Changed from bar_values
            barHeights={data.logits_hist_data_bar_heights} // Changed from bar_heights
          />
        </div>
        <div className="col-span-1"> {/* Set a fixed height to match the first div */}
          <LogitsTable
            negStr={data.neg_str}
            negValues={data.neg_values}
            posStr={data.pos_str}
            posValues={data.pos_values}
          />
        </div>
        <div className="col-span-2">
          <ActivationsDensity
            barValues={data.freq_hist_data_bar_values}
            barHeights={data.freq_hist_data_bar_heights}
          />
          <LogitDensity
            barValues={data.logits_hist_data_bar_values}
            barHeights={data.logits_hist_data_bar_heights}
          />
        </div>
      </div>
    </div>
  );
}