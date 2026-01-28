"use client";

import React, { useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Color, Fog, PerspectiveCamera, Scene, Vector3 } from "three";
import ThreeGlobe from "three-globe";

import countries from "@/data/globe.json";

const RING_PROPAGATION_SPEED = 3;
const ASPECT = 1.2;
const CAMERA_Z = 300;

export type Position = {
  order: number;
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  arcAlt: number;
  color: string;
};

export type GlobeConfig = {
  pointSize?: number;

  globeColor?: string;
  emissive?: string;
  emissiveIntensity?: number;
  shininess?: number;

  polygonColor?: string;

  showAtmosphere?: boolean;
  atmosphereColor?: string;
  atmosphereAltitude?: number;

  ambientLight?: string;
  directionalLeftLight?: string;
  directionalTopLight?: string;
  pointLight?: string;

  arcTime?: number;
  arcLength?: number;

  rings?: number;
  maxRings?: number;

  autoRotate?: boolean;
  autoRotateSpeed?: number;

  initialPosition?: { lat: number; lng: number };
};

export type WorldProps = {
  globeConfig?: GlobeConfig;
  data?: Position[];
};

/**
 * Small helper: ensures we always have a full config object.
 */
function useMergedConfig(globeConfig?: GlobeConfig) {
  return useMemo(
    () => ({
      pointSize: 1,

      showAtmosphere: true,
      atmosphereColor: "#ffffff",
      atmosphereAltitude: 0.1,

      polygonColor: "rgba(255,255,255,0.7)",

      globeColor: "#1d072e",
      emissive: "#000000",
      emissiveIntensity: 0.1,
      shininess: 0.9,

      arcTime: 2000,
      arcLength: 0.9,

      rings: 1,
      maxRings: 3,

      ambientLight: "#38bdf8",
      directionalLeftLight: "#ffffff",
      directionalTopLight: "#ffffff",
      pointLight: "#ffffff",

      autoRotate: true,
      autoRotateSpeed: 1,

      initialPosition: { lat: 0, lng: 0 },

      ...(globeConfig ?? {}),
    }),
    [globeConfig],
  );
}

function uniquePointsByLatLng<T extends { lat: number; lng: number }>(pts: T[]) {
  return pts.filter(
    (p, idx, arr) =>
      arr.findIndex((q) => q.lat === p.lat && q.lng === p.lng) === idx,
  );
}

function genRandomNumbers(min: number, max: number, count: number) {
  const out: number[] = [];
  const target = Math.max(0, Math.min(count, max - min));
  while (out.length < target) {
    const r = Math.floor(Math.random() * (max - min)) + min;
    if (!out.includes(r)) out.push(r);
  }
  return out;
}

function WebGLRendererConfig() {
  const { gl, size } = useThree();

  useEffect(() => {
    gl.setPixelRatio(window.devicePixelRatio);
    gl.setSize(size.width, size.height);
    gl.setClearColor(0x000000, 0);
  }, [gl, size]);

  return null;
}

function GlobeObject({
  globeConfig,
  data,
}: {
  globeConfig: ReturnType<typeof useMergedConfig>;
  data: Position[];
}) {
  const cfg = globeConfig;

  const groupRef = useRef<any>(null);
  const globeRef = useRef<ThreeGlobe | null>(null);

  // init once
  useEffect(() => {
    if (!groupRef.current || globeRef.current) return;
    globeRef.current = new ThreeGlobe();
    groupRef.current.add(globeRef.current);
  }, []);

  // material
  useEffect(() => {
    if (!globeRef.current) return;

    const mat = globeRef.current.globeMaterial() as unknown as {
      color: Color;
      emissive: Color;
      emissiveIntensity: number;
      shininess: number;
    };

    mat.color = new Color(cfg.globeColor);
    mat.emissive = new Color(cfg.emissive);
    mat.emissiveIntensity = cfg.emissiveIntensity ?? 0.1;
    mat.shininess = cfg.shininess ?? 0.9;
  }, [cfg.globeColor, cfg.emissive, cfg.emissiveIntensity, cfg.shininess]);

  // polygons + arcs + points + ring base
  useEffect(() => {
    if (!globeRef.current) return;
    if (!data.length) return;

    // points from arcs
    const points = data.flatMap((arc) => [
      {
        size: cfg.pointSize,
        order: arc.order,
        color: arc.color,
        lat: arc.startLat,
        lng: arc.startLng,
      },
      {
        size: cfg.pointSize,
        order: arc.order,
        color: arc.color,
        lat: arc.endLat,
        lng: arc.endLng,
      },
    ]);

    const filteredPoints = uniquePointsByLatLng(points);

    globeRef.current
      .hexPolygonsData((countries as any).features)
      .hexPolygonResolution(3)
      .hexPolygonMargin(0.7)
      .hexPolygonColor(() => cfg.polygonColor)
      .showAtmosphere(cfg.showAtmosphere)
      .atmosphereColor(cfg.atmosphereColor)
      .atmosphereAltitude(cfg.atmosphereAltitude);

    globeRef.current
      .arcsData(data)
      .arcStartLat((d) => (d as Position).startLat)
      .arcStartLng((d) => (d as Position).startLng)
      .arcEndLat((d) => (d as Position).endLat)
      .arcEndLng((d) => (d as Position).endLng)
      .arcColor((d) => (d as Position).color)
      .arcAltitude((d) => (d as Position).arcAlt)
      .arcStroke(() => [0.32, 0.28, 0.3][Math.round(Math.random() * 2)])
      .arcDashLength(cfg.arcLength)
      .arcDashInitialGap((d) => (d as Position).order)
      .arcDashGap(15)
      .arcDashAnimateTime(() => cfg.arcTime);

    globeRef.current
      .pointsData(filteredPoints as any)
      .pointColor((d: any) => d.color)
      .pointsMerge(true)
      .pointAltitude(0.0)
      .pointRadius(2);

    globeRef.current
      .ringsData([])
      .ringColor(() => cfg.polygonColor)
      .ringMaxRadius(cfg.maxRings)
      .ringPropagationSpeed(RING_PROPAGATION_SPEED)
      .ringRepeatPeriod((cfg.arcTime * cfg.arcLength) / Math.max(1, cfg.rings));
  }, [
    cfg.pointSize,
    cfg.polygonColor,
    cfg.showAtmosphere,
    cfg.atmosphereColor,
    cfg.atmosphereAltitude,
    cfg.arcLength,
    cfg.arcTime,
    cfg.rings,
    cfg.maxRings,
    data,
  ]);

  // rings animation
  useEffect(() => {
    if (!globeRef.current) return;
    if (!data.length) return;

    const interval = setInterval(() => {
      if (!globeRef.current) return;

      const picks = genRandomNumbers(
        0,
        data.length,
        Math.floor((data.length * 4) / 5),
      );

      const ringsData = data
        .filter((_, i) => picks.includes(i))
        .map((d) => ({
          lat: d.startLat,
          lng: d.startLng,
          color: d.color,
        }));

      globeRef.current.ringsData(ringsData as any);
    }, 2000);

    return () => clearInterval(interval);
  }, [data]);

  return <group ref={groupRef} />;
}

/**
 * Main component you render in your page:
 * <World globeConfig={...} data={...} />
 */
export function World({ globeConfig, data }: WorldProps) {
  const cfg = useMergedConfig(globeConfig);
  const arcs = data ?? [];

  const scene = useMemo(() => {
    const s = new Scene();
    s.fog = new Fog(0xffffff, 400, 2000);
    return s;
  }, []);

  const camera = useMemo(
    () => new PerspectiveCamera(50, ASPECT, 180, 1800),
    [],
  );

  return (
    <Canvas scene={scene} camera={camera}>
      <WebGLRendererConfig />

      <ambientLight color={cfg.ambientLight} intensity={0.6} />
      <directionalLight
        color={cfg.directionalLeftLight}
        position={new Vector3(-400, 100, 400)}
      />
      <directionalLight
        color={cfg.directionalTopLight}
        position={new Vector3(-200, 500, 200)}
      />
      <pointLight
        color={cfg.pointLight}
        position={new Vector3(-200, 500, 200)}
        intensity={0.8}
      />

      <GlobeObject globeConfig={cfg} data={arcs} />

      <OrbitControls
        enablePan={false}
        enableZoom={false}
        minDistance={CAMERA_Z}
        maxDistance={CAMERA_Z}
        autoRotate={cfg.autoRotate}
        autoRotateSpeed={cfg.autoRotateSpeed}
        minPolarAngle={Math.PI / 3.5}
        maxPolarAngle={Math.PI - Math.PI / 3}
      />
    </Canvas>
  );
}
