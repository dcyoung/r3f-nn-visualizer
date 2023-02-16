import * as tf from "@tensorflow/tfjs";
// @ts-ignore
import { TRAINING_DATA as ALL_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

export interface ModelConfig {
  inputSize: number;
  nClasses: number;
  hiddenLayerSizes: number[];
}
export const DEFAULT_MODEL_CONFIG = {
  inputSize: 28 * 28,
  nClasses: 10,
  hiddenLayerSizes: [16, 16],
};
export const getModel = (config: ModelConfig = DEFAULT_MODEL_CONFIG) => {
  const model = tf.sequential();

  // Hidden Layers
  model.add(
    tf.layers.dense({
      inputShape: [config.inputSize],
      units: config.hiddenLayerSizes[0],
      activation: "relu",
    })
  );
  config.hiddenLayerSizes.slice(1).forEach((hiddenLayerSize) => {
    model.add(
      tf.layers.dense({
        units: hiddenLayerSize,
        kernelInitializer: "varianceScaling",
        activation: "relu",
      })
    );
  });

  // Output layer
  model.add(
    tf.layers.dense({
      units: config.nClasses,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  return model;
};

export const getRandomSample = () => {
  // Input feature Array is 2 dimensional.
  const idx = Math.floor(Math.random() * ALL_DATA.outputs.length);
  const sample = ALL_DATA.inputs[idx];
  const label = ALL_DATA.outputs[idx];
  return [
    tf.tensor2d([sample], [1, sample.length]),
    tf.tensor1d([label], "int32"),
  ];
};

export const trainModel = async (
  model: tf.Sequential,
  nEpochs: number,
  onEpochEnd: (epoch: number, logs?: tf.Logs | undefined) => void
) => {
  // Grab a reference to the MNIST input values (pixel data)
  const INPUTS = ALL_DATA.inputs;
  // Grab reference to the MNIST output values.
  const OUTPUTS = ALL_DATA.outputs;

  // Shuffle the two arrays to remove any order, but do so in the same way so
  // inputs still match outputs indexes.
  tf.util.shuffleCombo(INPUTS, OUTPUTS);

  // Input feature Array is 2 dimensional.
  const INPUTS_TENSOR = tf.tensor2d(INPUTS);
  const OUTPUTS_TENSOR_ONE_HOT = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

  // Setup optimizer, loss function and accuracy metric...
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Train the model
  let history = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR_ONE_HOT, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: nEpochs,
    callbacks: {
      onEpochEnd: onEpochEnd,
    },
  });

  INPUTS_TENSOR.dispose();
  OUTPUTS_TENSOR_ONE_HOT.dispose();

  return history;
};
