import "./App.css";
import { Suspense, useMemo, useState } from "react";
import { DEFAULT_MODEL_CONFIG, getModel } from "./model";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import ActivationVisual from "./ActivationVisual";
import ModelManager from "./ModelManager";

const App = () => {
  const modelConfig = DEFAULT_MODEL_CONFIG;
  const model = useMemo(() => {
    const m = getModel(modelConfig);
    return m;
  }, [modelConfig]);
  const [showSynapses, setShowSynapses] = useState(true);

  const backgroundColor = "#282828";
  return (
    <Suspense fallback={<span>loading...</span>}>
      <ModelManager model={model} />
      <div id="info" style={{ left: "0.5em", bottom: "0.5em" }}>
        <button
          onClick={() => {
            setShowSynapses((prev) => !prev);
          }}
        >
          Synapse {showSynapses ? "☑" : "☐"}
        </button>
      </div>
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
        <ambientLight />
        <ActivationVisual
          modelConfig={modelConfig}
          showSynapses={showSynapses}
        />
        <OrbitControls makeDefault />
      </Canvas>
    </Suspense>
  );
};

export default App;
