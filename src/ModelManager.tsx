import * as tf from "@tensorflow/tfjs";
import { useState } from "react";
import ActivationProbe from "./ActivationProbe";
import { trainModel } from "./model";
import ProgressBar from "./ProgressBar";

class EpochLog {
  epoch: number;
  nEpochs: number;
  logs: tf.Logs | undefined;

  constructor(epoch: number, nEpochs: number, logs?: tf.Logs | undefined) {
    this.epoch = epoch;
    this.nEpochs = nEpochs;
    this.logs = logs;
  }

  toString(): string {
    return (
      "Data for epoch " +
      this.epoch +
      ", " +
      JSON.stringify(this.logs, function (key, val) {
        return val.toFixed ? Number(val.toFixed(3)) : val;
      })
    );
  }
}

interface ModelManagerProps {
  model: tf.Sequential;
}
const ModelManager = ({ model, ...props }: ModelManagerProps) => {
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);
  const [modelValAcc, setModelValAcc] = useState("NA");
  const nEpochs = 50;
  const [trained, setTrained] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

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
  };

  return (
    <div
      style={{
        width: "100%",
        height: "10%",
        display: "flex",
        flexDirection: "row",
        justifyContent: "left",
        gap: "20px",
        alignItems: "center",
      }}
    >
      <ActivationProbe model={model} disabled={isTraining} />
      {trained ? (
        <p>Using trained model w/ final accuracy {modelValAcc}</p>
      ) : isTraining ? (
        <>
          <ProgressBar
            title="Training"
            completed={(100 * epochLogs.length) / nEpochs}
          />
          <p>Val Acc: {modelValAcc}</p>
        </>
      ) : (
        <>
          <button
            style={{ height: "75%" }}
            disabled={trained}
            onClick={async () => {
              setIsTraining(true);
              const history = await trainModel(model, nEpochs, onEpochEnd);
              onTrainingComplete(history);
            }}
          >
            Train Model
          </button>
          <div className="hideScroll">
            <p>Activations will be random. Try training the model.</p>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelManager;
