import * as tf from "@tensorflow/tfjs";
import { useState } from "react";
import { trainModel } from "./model";

interface ModelTrainerProps {
  model: tf.Sequential;
}

class EpochLog {
  epoch: number;
  logs: tf.Logs | undefined;

  constructor(epoch: number, logs?: tf.Logs | undefined) {
    this.epoch = epoch;
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

const ModelTrainer = ({ model, ...props }: ModelTrainerProps): JSX.Element => {
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);

  const onEpochEnd = (epoch: number, logs?: tf.Logs | undefined) => {
    // console.log("Data for epoch " + epoch, logs);
    setEpochLogs((oldArray) => [new EpochLog(epoch, logs), ...oldArray]);
  };

  const onTrainingComplete = (history: tf.History) => {
    console.log(history);
  };

  return (
    <div>
      <button
        onClick={async () => {
          const history = await trainModel(model, onEpochEnd);
          onTrainingComplete(history);
        }}
      >
        Train Model!
      </button>
      <div style={{ height: "100px", overflowY: "scroll" }}>
        {epochLogs.map((elem, i) => (
          <p key={i}>{elem.toString()}</p>
        ))}
      </div>
    </div>
  );
};

export default ModelTrainer;
