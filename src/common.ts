export interface ModelConfig {
  inputSize: number;
  nClasses: number;
  hiddenLayerSizes: number[];
}

const countNeurons = (modelConfig: ModelConfig): number => {
  return (
    modelConfig.inputSize +
    modelConfig.hiddenLayerSizes.reduce((prev, curr) => prev + curr) +
    modelConfig.nClasses
  );
};

const countSynapses = (modelConfig: ModelConfig): number => {
  let total = 0;
  const layerSizes = [...modelConfig.hiddenLayerSizes, modelConfig.nClasses];
  layerSizes.forEach((n, i) => {
    const inputSize = i == 0 ? modelConfig.inputSize : layerSizes[i - 1];
    total += n * inputSize;
  });
  return total;
};

export class ActivationData {
  public readonly modelInput: Float32Array | Uint8Array | Int32Array;
  public readonly modelActivations: (Float32Array | Uint8Array | Int32Array)[];
  constructor(
    modelInput: Float32Array | Uint8Array | Int32Array,
    modelActivations: (Float32Array | Uint8Array | Int32Array)[]
  ) {
    this.modelInput = modelInput;
    this.modelActivations = modelActivations;
  }
}

export class ModelActivationsHelper {
  public readonly modelConfig: ModelConfig;
  public get layerSizes() {
    return [
      this.modelConfig.inputSize,
      ...this.modelConfig.hiddenLayerSizes,
      this.modelConfig.nClasses,
    ];
  }
  public get nLayers() {
    return this.layerSizes.length;
  }
  public get layerStartIdxs() {
    const out = [0];
    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      out.push(out[i] + this.layerSizes[i]);
    }
    return out;
  }

  public get nNeurons() {
    return countNeurons(this.modelConfig);
  }
  public get nSynapses() {
    return countSynapses(this.modelConfig);
  }

  public get input2DSideLength() {
    return Math.sqrt(this.modelConfig.inputSize);
  }

  public get maxNeuronsX() {
    return Math.max(...[this.input2DSideLength, ...this.layerSizes.slice(1)]);
  }

  public get maxNeuronsY() {
    return this.nLayers;
  }

  public getNeuronsByLayer(layerIdx: number) {
    const nextLayerIdx = layerIdx + 1;
    const start = this.layerStartIdxs[layerIdx];
    const stop =
      nextLayerIdx < this.nLayers
        ? start + this.layerSizes[layerIdx]
        : this.nNeurons;
    return Array.from({ length: stop - start }, (_, i) => start + i);
  }

  public getLayerIdx(nIdx: number) {
    return findLastIndex(this.layerStartIdxs, (v) => v <= nIdx);
  }

  public getActivation(data: ActivationData, nIdx: number) {
    const layerIdx = this.getLayerIdx(nIdx);
    if (layerIdx == 0) {
      return data.modelInput[nIdx];
    }
    const localNIdx = nIdx - this.layerStartIdxs[layerIdx];
    return data.modelActivations[layerIdx - 1][localNIdx];
  }

  public getNeuronXYZNorm(nIdx: number) {
    const layerIdx = this.getLayerIdx(nIdx);
    // Get's xyz values (in range -1:1) for neuron by index
    if (layerIdx == 0) {
      const row = Math.floor(nIdx / this.input2DSideLength);
      const col = nIdx % this.input2DSideLength;
      return [
        (2 * col) / this.input2DSideLength - 1,
        (2 * layerIdx) / (this.nLayers - 1) - 1,
        1 - (2 * row) / this.input2DSideLength,
      ];
    }

    const localNIdx = nIdx - this.layerStartIdxs[layerIdx];
    return [
      (2 * localNIdx) / this.layerSizes[layerIdx] - 1,
      (2 * layerIdx) / (this.nLayers - 1) - 1,
      0,
    ];
  }

  public getNSynapsesForNeuron(nIdx: number) {
    const layerIdx = this.getLayerIdx(nIdx);
    const nextLayerIdx = layerIdx + 1;
    if (nextLayerIdx >= this.nLayers) {
      // output layer... no further connections
      return 0;
    }
    return this.layerSizes[nextLayerIdx];
  }

  public getNSynapsesForLayer(layerIdx: number) {
    return (
      this.layerSizes[layerIdx] *
      this.getNSynapsesForNeuron(this.layerStartIdxs[layerIdx])
    );
  }

  public getDstNeuronIdxs(nIdx: number) {
    const layerIdx = this.getLayerIdx(nIdx);
    const nextLayerIdx = layerIdx + 1;
    if (nextLayerIdx >= this.nLayers) {
      // output layer... no further connections
      return [];
    }

    return this.getNeuronsByLayer(nextLayerIdx);
  }

  private getSynapseMap() {
    return Array.from({ length: this.nNeurons }).map((_, i) =>
      this.getDstNeuronIdxs(i)
    );
  }

  public getFlattenedSynapseMap() {
    return this.getSynapseMap()
      .map((dsts, src) =>
        dsts.map((dst) => {
          return { srcIdx: src, dstIdx: dst };
        })
      )
      .reduce((a, b) => a.concat(b), []);
  }

  constructor(modelConfig: ModelConfig) {
    this.modelConfig = modelConfig;
  }
}

/**
 * Returns the index of the last element in the array where predicate is true, and -1
 * otherwise.
 * @param array The source array to search in
 * @param predicate find calls predicate once for each element of the array, in descending
 * order, until it finds one where predicate returns true. If such an element is found,
 * findLastIndex immediately returns that element index. Otherwise, findLastIndex returns -1.
 */
export function findLastIndex<T>(
  array: Array<T>,
  predicate: (value: T, index: number, obj: T[]) => boolean
): number {
  let l = array.length;
  while (l--) {
    if (predicate(array[l], l, array)) return l;
  }
  return -1;
}
