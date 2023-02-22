import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";
import { useAppStateActions } from "./appState";
import {
  getRandomSample,
  trainModel,
  probeModelActivation,
  EpochLog,
} from "./model";
import ProgressBar from "./ProgressBar";
import "./Overlay.css";

interface ModelManagerProps {
  model: tf.Sequential;
}
const ModelManager = ({ model, ...props }: ModelManagerProps) => {
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);
  const [modelValAcc, setModelValAcc] = useState("NA");
  const nEpochs = 50;
  const [trained, setTrained] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

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

  const onEpochEnd = (epoch: number, logs?: tf.Logs | undefined) => {
    // console.log("Data for epoch " + epoch, logs);
    setModelValAcc(logs ? `${Math.floor(100 * logs.val_acc)}%` : "NA");
    setEpochLogs((oldArray) => [
      new EpochLog(epoch, nEpochs, logs),
      ...oldArray,
    ]);
  };

  const onTrainingComplete = (history: tf.History) => {
    console.log(history);
    setTrained(true);
    setIsTraining(false);
    const acc = history.history.val_acc.slice(-1)[0] as number;
    setModelValAcc(`${Math.floor(100 * acc)}%`);
    onGetActivation();
  };

  return (
    <>
      <div
        id="info"
        style={{
          top: "0em",
          width: "100%",
          textAlign: "right",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            display: "inline",
          }}
        />
      </div>
      <div id="info" style={{ top: "0.5em", left: "0.5em" }}>
        {trained ? (
          <p>Using trained model</p>
        ) : isTraining ? (
          <div style={{ width: "50vw" }}>
            <p>Training model...</p>
            <ProgressBar
              title=""
              completed={(100 * epochLogs.length) / nEpochs}
            />
          </div>
        ) : (
          <>
            <p>Using untrained model.</p>
            <button
              disabled={trained}
              onClick={async () => {
                setIsTraining(true);
                const history = await trainModel(model, nEpochs, onEpochEnd);
                onTrainingComplete(history);
              }}
            >
              Train Model
            </button>
          </>
        )}
      </div>
      <div id="info" style={{ bottom: "0.5em", right: "0.5em" }}>
        <button disabled={isTraining} onClick={onGetActivation}>
          New Activation
        </button>
      </div>
    </>
  );
};

export default ModelManager;
