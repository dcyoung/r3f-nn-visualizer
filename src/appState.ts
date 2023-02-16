import create from "zustand";
import { ActivationData } from "./common";

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
