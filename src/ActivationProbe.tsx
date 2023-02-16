import * as tf from "@tensorflow/tfjs";
import { useRef } from "react";
import { ActivationData, useAppStateActions } from "./appState";
import { getRandomSample } from "./model";

interface ActivationProbeProps {
  model: tf.Sequential;
}

const ActivationProbe = ({
  model,
  ...props
}: ActivationProbeProps): JSX.Element => {
  const canvasRef = useRef<HTMLCanvasElement>(null!);

  const { updateActivationData } = useAppStateActions();

  const probeModelActivation = async () => {
    const [sample, label] = getRandomSample();
    if (canvasRef.current) {
      const imgTensor: tf.Tensor3D = sample.reshape([28, 28, 1]);
      await tf.browser.toPixels(imgTensor, canvasRef.current);
    }

    // const layerInputs_BK = [tf.zeros([1, 28 * 28 * 1])];
    const layerInputs: tf.Tensor<tf.Rank>[] = [sample];
    model.layers.forEach((layer, i) => {
      const layerOutput = layer.apply(layerInputs[i]);
      layerInputs.push(layerOutput as tf.Tensor<tf.Rank>);
    });
    return new ActivationData(
      await layerInputs[0].data(),
      await Promise.all(layerInputs.slice(1).map(async (t) => await t.data()))
    );
  };

  return (
    <>
      <div>
        <canvas
          ref={canvasRef}
          width={28}
          height={28}
          style={{ margin: "4px" }}
        ></canvas>
        ;
        <button
          onClick={async () => {
            const results = await probeModelActivation();
            updateActivationData(results);
          }}
        >
          Probe Model Activation!
        </button>
      </div>
    </>
  );
};

export default ActivationProbe;
