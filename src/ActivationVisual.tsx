import { useEffect, useMemo, useRef, useState } from "react";
import {
  BoxGeometry,
  Color,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  Vector2,
} from "three";
import { Lut } from "three/examples/jsm/math/Lut.js";
import { Line2, LineSegmentsGeometry, LineMaterial } from "three-stdlib";
import { useActivationData } from "./appState";
import { ModelConfig } from "./model";
import { ModelActivationsHelper } from "./common";
import { useFrame } from "@react-three/fiber";

interface ActivationVisualProps {
  modelConfig: ModelConfig;
}

const ActivationVisual = ({ modelConfig, ...props }: ActivationVisualProps) => {
  const helper = new ModelActivationsHelper(modelConfig);
  const visualizePropagation = false;
  const visualizePropagationSweep = false;
  const forwardPropSec = 5;
  const cubeSideLength = 0.35;
  const cubeSpacingScalar = 1;
  const boundsX = cubeSpacingScalar * cubeSideLength;
  const boundsY = 40 * cubeSpacingScalar * cubeSideLength;
  const boundsZ = cubeSpacingScalar * cubeSideLength;
  const neuronGeo = useMemo(() => {
    return new BoxGeometry(cubeSideLength, cubeSideLength, cubeSideLength, 1);
  }, []);
  const neuronMat = useMemo(() => {
    return new MeshBasicMaterial({ color: "white", toneMapped: false });
  }, []);

  const [lineMat] = useState(
    () =>
      new LineMaterial({
        linewidth: 0.15,
        vertexColors: true,
        opacity: 0.8,
        resolution: new Vector2(512, 512),
      })
  );
  const [lineGeo] = useState(() => new LineSegmentsGeometry());
  const [lineObj] = useState(() => new Line2());
  const [linePositions] = useState<number[]>(() =>
    Array(helper.nSynapses * 6).fill(0)
  );
  const [lineColors] = useState<Float32Array>(
    () => new Float32Array(Array(helper.nSynapses * 8).fill(0))
  );

  const data = useActivationData();
  const lut = new Lut("cooltowarm").setMin(0).setMax(1);
  const meshRef = useRef<InstancedMesh>(null!);
  const tmpMatrix = useMemo(() => new Matrix4(), []);

  const updateSynapseColors = (
    normPropagationProgress: number = 0,
    sweep: boolean = true
  ) => {
    let srcIdx = 0;
    let [r, g, b] = [0, 0, 0];
    let [r2, g2, b2] = [0, 0, 0];
    let tmpColor = new Color();
    let cUnreached = new Color("black");
    let prevSrcIdx = -1;
    let normLayerIdx = -1;
    let delta = -1;
    const normStepBetweenLayers = 1 / (helper.nLayers - 1);
    const flattened = helper.getFlattenedSynapseMap();
    for (let sIdx = 0; sIdx < flattened.length; sIdx++) {
      srcIdx = flattened[sIdx].srcIdx;
      if (srcIdx !== prevSrcIdx) {
        prevSrcIdx = srcIdx;
        normLayerIdx = helper.getLayerIdx(srcIdx) / (helper.nLayers - 1);

        delta = normPropagationProgress - normLayerIdx;
        if (delta >= 0) {
          // propagation has passed this layer... we should use activation to color
          [r, g, b] = lut
            .getColor(data ? helper.getActivation(data, srcIdx) : 0)
            .toArray();

          if (sweep && delta >= 2 * normStepBetweenLayers) {
            // propagation is far beyond this layer... deactivate it for a sweeping look
            [r2, g2, b2] = tmpColor
              .setRGB(r, g, b)
              .lerp(
                cUnreached,
                (delta % normStepBetweenLayers) / normStepBetweenLayers
              );
            [r, g, b] = cUnreached.toArray();
          } else if (delta >= normStepBetweenLayers) {
            // propagation has passed the next layer... color it the same
            [r2, g2, b2] = [r, g, b];
          } else {
            // propagation has NOT reached the next layer... create a gradient
            [r2, g2, b2] = tmpColor
              .setRGB(r, g, b)
              .lerp(cUnreached, delta / normStepBetweenLayers);
          }
        } else {
          // Both are unreached
          [r, g, b] = cUnreached.toArray();
          [r2, g2, b2] = cUnreached.toArray();
        }
      }
      lineColors[sIdx * 6 + 0] = r;
      lineColors[sIdx * 6 + 1] = g;
      lineColors[sIdx * 6 + 2] = b;
      lineColors[sIdx * 6 + 3] = r2;
      lineColors[sIdx * 6 + 4] = g2;
      lineColors[sIdx * 6 + 5] = b2;
      lineGeo.attributes.color.setX(sIdx + 0, r);
      lineGeo.attributes.color.setX(sIdx + 1, g);
      lineGeo.attributes.color.setX(sIdx + 2, b);
      lineGeo.attributes.color.setX(sIdx + 3, r2);
      lineGeo.attributes.color.setX(sIdx + 4, g2);
      lineGeo.attributes.color.setX(sIdx + 5, b2);
    }
    lineGeo.setColors(lineColors);
    lineGeo.attributes.color.needsUpdate = true;
  };

  const updateSynapsePositions = () => {
    let srcIdx = 0,
      dstIdx = 0;
    const flattened = helper.getFlattenedSynapseMap();
    for (let sIdx = 0; sIdx < flattened.length; sIdx++) {
      srcIdx = flattened[sIdx].srcIdx;
      dstIdx = flattened[sIdx].dstIdx;
      const [x, y, z] = helper.getNeuronXYZNorm(srcIdx);
      const [x2, y2, z2] = helper.getNeuronXYZNorm(dstIdx);
      const [sX, sY, sZ] = helper.getNeuronCountXYZForLayer(
        helper.getLayerIdx(srcIdx)
      );
      const [sX2, sY2, sZ2] = helper.getNeuronCountXYZForLayer(
        helper.getLayerIdx(dstIdx)
      );
      linePositions[sIdx * 6 + 0] = x * sX * boundsX;
      linePositions[sIdx * 6 + 1] = y * sY * boundsY;
      linePositions[sIdx * 6 + 2] = z * sZ * boundsZ;
      linePositions[sIdx * 6 + 3] = x2 * sX2 * boundsX;
      linePositions[sIdx * 6 + 4] = y2 * sY2 * boundsY;
      linePositions[sIdx * 6 + 5] = z2 * sZ2 * boundsZ;
    }
    lineGeo.setPositions(linePositions);
    lineObj.computeLineDistances();
  };

  useFrame(({ clock }) => {
    const normProgress = visualizePropagation
      ? (clock.getElapsedTime() % forwardPropSec) / forwardPropSec
      : 1.0;
    // update synapses
    updateSynapseColors(normProgress, visualizePropagationSweep);

    // update neurons
    // for (let nIdx = 0; nIdx < helper.nNeurons; nIdx++) {
    //   const normLayerIdx = helper.getLayerIdx(nIdx) / (helper.nLayers - 1);
    //   if (normLayerIdx <= normProgress) {
    //     const act = data ? helper.getActivation(data, nIdx) : 0;
    //     meshRef.current.setColorAt(nIdx, lut.getColor(act));
    //   } else {
    //     meshRef.current.setColorAt(nIdx, new Color("black"));
    //   }
    // }
    // meshRef.current.instanceColor!.needsUpdate = true;
  });

  useEffect(() => {
    // Setup Neuron Positions and Colors
    for (let nIdx = 0; nIdx < helper.nNeurons; nIdx++) {
      let [x, y, z] = helper.getNeuronXYZNorm(nIdx);
      const [sX, sY, sZ] = helper.getNeuronCountXYZForLayer(
        helper.getLayerIdx(nIdx)
      );
      tmpMatrix.setPosition(
        x * sX * boundsX,
        y * sY * boundsY,
        z * sZ * boundsZ
      );
      meshRef.current.setMatrixAt(nIdx, tmpMatrix);

      const act = data ? helper.getActivation(data, nIdx) : 0;
      meshRef.current.setColorAt(nIdx, lut.getColor(act));
    }

    meshRef.current.instanceMatrix!.needsUpdate = true;
    meshRef.current.instanceColor!.needsUpdate = true;

    // Setup synapses
    updateSynapsePositions();
    updateSynapseColors();
  }, [helper, meshRef, lineObj]);

  useEffect(() => {
    if (!data) {
      return;
    }

    // Update neuron colors
    for (let nIdx = 0; nIdx < helper.nNeurons; nIdx++) {
      const act = data ? helper.getActivation(data, nIdx) : 0;
      meshRef.current.setColorAt(nIdx, lut.getColor(act));
    }
    meshRef.current.instanceColor!.needsUpdate = true;
  }, [helper, data]);

  return (
    <>
      <instancedMesh
        ref={meshRef}
        castShadow={true}
        receiveShadow={true}
        args={[neuronGeo, neuronMat, helper.nNeurons]}
      />
      <primitive object={lineObj}>
        <primitive object={lineGeo} attach="geometry">
          <bufferAttribute
            attach="attributes-color"
            count={helper.nSynapses}
            array={lineColors}
            itemSize={lineColors.length / helper.nSynapses}
          />
        </primitive>
        <primitive
          object={lineMat}
          attach="material"
          vertexColors={true}
          // resolution={new Vector2(512, 512)}
          // lineWidth={0.1}
        />
      </primitive>
    </>
  );
};

export default ActivationVisual;
