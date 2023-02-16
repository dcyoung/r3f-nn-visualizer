import "./App.css";
import { Suspense, useMemo } from "react";
import { DEFAULT_MODEL_CONFIG, getModel } from "./model";
import ModelTrainer from "./ModelTrainer";
import ActivationProbe from "./ActivationProbe";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import ActivationVisual from "./ActivationVisual";

const App = (): JSX.Element => {
  const modelConfig = DEFAULT_MODEL_CONFIG;
  const model = useMemo(() => {
    const m = getModel(modelConfig);
    // console.log(m.summary());
    return m;
  }, [modelConfig]);

  const backgroundColor = "#010204";
  // const backgroundColor = "#A9DFBF";
  return (
    <Suspense fallback={<span>loading...</span>}>
      <ModelTrainer model={model} />
      <ActivationProbe model={model} />
      <Canvas
        camera={{
          fov: 45,
          near: 1,
          far: 1000,
          position: [-17, -6, 6.5],
          up: [0, 0, 1],
        }}
      >
        <color attach="background" args={[backgroundColor]} />
        <ambientLight />
        <ActivationVisual modelConfig={modelConfig} />
        <OrbitControls makeDefault />
      </Canvas>
    </Suspense>
  );
};

export default App;
