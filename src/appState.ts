import create from "zustand";

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

interface IAppState {
  activationData: ActivationData | null;
  actions: {
    updateActivationData: (newData: ActivationData) => void;
  };
}

const useAppState = create<IAppState>((set, get) => ({
  activationData: null,
  actions: {
    updateActivationData: (newData: ActivationData) =>
      set((state) => {
        return {
          activationData: newData,
        };
      }),
  },
}));

export const useActivationData = () =>
  useAppState((state) => state.activationData);
export const useAppStateActions = () => useAppState((state) => state.actions);
