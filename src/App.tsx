import "./App.css";
import { Suspense, useMemo } from "react";
import { DEFAULT_MODEL_CONFIG, getModel } from "./model";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import ActivationVisual from "./ActivationVisual";
import ModelManager from "./ModelManager";

const App = () => {
  const modelConfig = DEFAULT_MODEL_CONFIG;
  const model = useMemo(() => {
    const m = getModel(modelConfig);
    // console.log(m.summary());
    return m;
  }, [modelConfig]);

  //const backgroundColor = "#010204";
  const backgroundColor = "#282828";
  return (
    <Suspense fallback={<span>loading...</span>}>
      <ModelManager model={model} />
      <Canvas
        camera={{
          fov: 45,
          near: 1,
          far: 1000,
          position: [-40, -45, 8],
          up: [0, 0, 1],
        }}
      >
        <color attach="background" args={[backgroundColor]} />
        {/* <axesHelper /> */}
        <ambientLight />
        <ActivationVisual modelConfig={modelConfig} />
        <OrbitControls makeDefault />
      </Canvas>
    </Suspense>
  );
};

export default App;
