"use client";

import { World } from "../components/ui/globe";
import globeData from "../data/globe.json";

export default function Home() {
  return (
    <main className="min-h-screen bg-black flex items-center justify-center">
      <div className="h-[600px] w-[600px]">
        <World
          data={globeData as any}
          globeConfig={{
            globeColor: "#0b1020",
            emissiveIntensity: 0.15,
            autoRotate: true,
            autoRotateSpeed: 0.9,
          }}
        />
      </div>
    </main>
  );
}
