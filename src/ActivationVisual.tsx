import { useEffect, useMemo, useRef, useState } from "react";
import {
  BoxGeometry,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  Vector2,
} from "three";
import { Lut } from "three/examples/jsm/math/Lut.js";
import { Line2, LineSegmentsGeometry, LineMaterial } from "three-stdlib";
import { ActivationData, useActivationData } from "./appState";
import { ModelConfig } from "./model";
import { findLastIndex } from "./common";

interface ActivationVisualProps {
  modelConfig: ModelConfig;
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

class ModelActivationsHelper {
  public readonly modelConfig: ModelConfig;
  public get layerSizes() {
    return [
      this.modelConfig.inputSize,
      ...this.modelConfig.hiddenLayerSizes,
      this.modelConfig.nClasses,
    ];
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

  public get maxX() {
    return Math.max(...[this.input2DSideLength, ...this.layerSizes.slice(1)]);
  }

  public get maxY() {
    return this.layerSizes.length;
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
        (2 * layerIdx) / this.layerSizes.length - 1,
        1 - (2 * row) / this.input2DSideLength,
      ];
    }

    const localNIdx = nIdx - this.layerStartIdxs[layerIdx];
    return [
      (2 * localNIdx) / this.layerSizes[layerIdx] - 1,
      (2 * layerIdx) / this.layerSizes.length - 1,
      0,
    ];
  }

  public getDstNeuronIdxs(nIdx: number) {
    const layerIdx = this.getLayerIdx(nIdx);
    const nextLayerIdx = layerIdx + 1;
    if (nextLayerIdx >= this.layerSizes.length) {
      // output layer... no further connections
      return [];
    }

    const start = this.layerStartIdxs[nextLayerIdx];
    const end = start + this.layerSizes[nextLayerIdx];
    return Array.from({ length: this.nNeurons })
      .map((_, i) => i)
      .slice(start, end);
  }

  public getSynapseMap() {
    return Array.from({ length: this.nNeurons }).map((_, i) =>
      this.getDstNeuronIdxs(i)
    );
  }

  constructor(modelConfig: ModelConfig) {
    this.modelConfig = modelConfig;
  }
}

const ActivationVisual = ({
  modelConfig,
  ...props
}: ActivationVisualProps): JSX.Element => {
  const helper = new ModelActivationsHelper(modelConfig);
  const cubeSideLength = 0.35;
  const cubeSpacingScalar = 1;
  const pixBounds =
    helper.input2DSideLength * cubeSpacingScalar * cubeSideLength;
  const boundsX = helper.maxX * cubeSpacingScalar * cubeSideLength;
  const boundsY = 20 * helper.maxY * cubeSpacingScalar * cubeSideLength;
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

  const updateSynapseColors = () => {
    let sIdx = 0;
    helper.getSynapseMap().forEach((dstIdxs, srcIdx) => {
      const c = lut.getColor(data ? helper.getActivation(data, srcIdx) : 0);
      dstIdxs.forEach((dstIdx) => {
        lineColors[sIdx * 6 + 0] = c.r;
        lineColors[sIdx * 6 + 1] = c.g;
        lineColors[sIdx * 6 + 2] = c.b;
        lineColors[sIdx * 6 + 3] = c.r;
        lineColors[sIdx * 6 + 4] = c.g;
        lineColors[sIdx * 6 + 5] = c.b;
        lineGeo.attributes.color.setX(sIdx, c.r);
        lineGeo.attributes.color.setX(sIdx + 1, c.g);
        lineGeo.attributes.color.setX(sIdx + 2, c.b);
        lineGeo.attributes.color.setX(sIdx + 3, c.r);
        lineGeo.attributes.color.setX(sIdx + 4, c.g);
        lineGeo.attributes.color.setX(sIdx + 5, c.b);
        sIdx += 1;
      });
    });
    lineGeo.setColors(lineColors);
    lineGeo.attributes.color.needsUpdate = true;
  };

  const updateSynapsePositions = () => {
    let sIdx = 0;
    helper.getSynapseMap().forEach((dstIdxs, srcIdx) => {
      dstIdxs.forEach((dstIdx) => {
        let [x, y, z] = helper.getNeuronXYZNorm(srcIdx);
        x *= helper.getLayerIdx(srcIdx) == 0 ? pixBounds : boundsX;
        y *= boundsY;
        z *= helper.getLayerIdx(srcIdx) == 0 ? pixBounds : 1;
        let [x2, y2, z2] = helper.getNeuronXYZNorm(dstIdx);
        x2 *= helper.getLayerIdx(dstIdx) == 0 ? pixBounds : boundsX;
        y2 *= boundsY;
        z2 *= helper.getLayerIdx(dstIdx) == 0 ? pixBounds : 1;
        linePositions[sIdx * 6 + 0] = x;
        linePositions[sIdx * 6 + 1] = y;
        linePositions[sIdx * 6 + 2] = z;
        linePositions[sIdx * 6 + 3] = x2;
        linePositions[sIdx * 6 + 4] = y2;
        linePositions[sIdx * 6 + 5] = z2;
        sIdx += 1;
      });
    });
    lineGeo.setPositions(linePositions);
    lineObj.computeLineDistances();
  };

  useEffect(() => {
    // Setup Neuron Positions and Colors
    for (let nIdx = 0; nIdx < helper.nNeurons; nIdx++) {
      let [x, y, z] = helper.getNeuronXYZNorm(nIdx);
      x *= helper.getLayerIdx(nIdx) == 0 ? pixBounds : boundsX;
      y *= boundsY;
      z *= helper.getLayerIdx(nIdx) == 0 ? pixBounds : 1;
      tmpMatrix.setPosition(x, y, z);
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

    // Update synapses colors
    updateSynapseColors();
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
