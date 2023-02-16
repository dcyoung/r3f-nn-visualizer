import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef } from "react";
import { useAppStateActions } from "./appState";
import { ActivationData } from "./common";
import { getRandomSample } from "./model";

const probeModelActivation = async (
  sample: tf.Tensor<tf.Rank>,
  model: tf.Sequential
) => {
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

interface ActivationProbeProps {
  model: tf.Sequential;
  disabled?: boolean;
}

const ActivationProbe = ({
  model,
  disabled = false,
  ...props
}: ActivationProbeProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null!);

  const { updateActivationData } = useAppStateActions();
  const onGetActivation = async () => {
    const [sample, label] = getRandomSample();
    if (canvasRef.current) {
      const imgTensor: tf.Tensor3D = sample.reshape([28, 28, 1]);
      await tf.browser.toPixels(imgTensor, canvasRef.current);
    }
    const results = await probeModelActivation(sample, model);
    updateActivationData(results);
  };

  useEffect(() => {
    onGetActivation();
  }, []);

  return (
    <>
      <button
        disabled={disabled}
        style={{ background: "#1e88e5" }}
        onClick={onGetActivation}
      >
        New Activations
      </button>
      <canvas
        ref={canvasRef}
        width={28}
        height={28}
        style={{ margin: "4px" }}
      />
    </>
  );
};

export default ActivationProbe;
