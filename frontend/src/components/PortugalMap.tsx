import React, { Component, ReactNode, useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export interface Hospital {
  name: string;
  lat: number;
  lon: number;
  region: string;
  consultations?: number; // usado como valor da métrica atual no heatmap
}

interface PortugalMapProps {
  hospitals: Hospital[];
  onHospitalClick?: (hospital: Hospital) => void;
  showHeatmap?: boolean;
  metric?: "consultations" | "carbapenems";
  maxReference?: number;
}

class MapErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown, info: React.ErrorInfo) {
    console.error("Erro ao renderizar PortugalMap:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex items-center justify-center bg-muted rounded-md border">
          <div className="text-center text-sm text-muted-foreground">
            <p>Não foi possível carregar o mapa interactivo.</p>
            <p className="mt-1">Verifique a configuração do react-leaflet.</p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function getColor(intensity: number): string {
  const t = Math.max(0, Math.min(1, intensity));

  const start = { r: 250, g: 204, b: 21 }; // amarelo
  const end = { r: 220, g: 38, b: 38 }; // vermelho

  const r = Math.round(start.r + (end.r - start.r) * t);
  const g = Math.round(start.g + (end.g - start.g) * t);
  const b = Math.round(start.b + (end.b - start.b) * t);

  const toHex = (v: number) => v.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function InnerPortugalMap({ hospitals, showHeatmap = true, metric = "consultations", maxReference }: PortugalMapProps) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const leafletMapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    if (!leafletMapRef.current) {
      leafletMapRef.current = L.map(mapRef.current).setView([39.5, -8.0], 6);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
          "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
      }).addTo(leafletMapRef.current);
    }

    const map = leafletMapRef.current;
    if (!map) return;

    map.eachLayer((layer) => {
      if ((layer as any).options && !(layer as any).getAttribution) {
        map.removeLayer(layer);
      }
    });

    const maxValue =
      (maxReference && maxReference > 0)
        ? maxReference
        : Math.max(1, ...hospitals.map((h) => h.consultations || 0));

    if (showHeatmap) {
      hospitals.forEach((hospital) => {
        const value = hospital.consultations || 0;
        const intensity = Math.min(1, value / maxValue); // clampar entre 0-1
        const radius = 8 + 18 * intensity; // raio fixo entre 8-26px baseado na intensidade normalizada
        const color = getColor(intensity);

        const circle = L.circleMarker([hospital.lat, hospital.lon], {
          radius,
          color: "transparent",
          fillColor: color,
          fillOpacity: 0.6,
        }).addTo(map);

        const formattedValue = value.toLocaleString("pt-PT");
        const unitLabel = metric === "carbapenems" ? "unidades de carbapenemes" : "consultas";

        const popupContent = `
          <div style="font-size: 11px;">
            <div style="font-weight: 600;">${hospital.name}</div>
            <div style="font-size: 10px; color: #4b5563;">${hospital.region}</div>
            <div style="font-size: 10px; margin-top: 4px;">${formattedValue} ${unitLabel}</div>
          </div>
        `;

        circle.bindPopup(popupContent);
      });
    }

    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
    };
  }, [hospitals, showHeatmap, metric, maxReference]);

  return (
    <div className="relative w-full h-full">
      <div
        ref={mapRef}
        style={{ height: "100%", width: "70%", minHeight: "500px" }}
        className="rounded-lg overflow-hidden border border-gray-200"
      />

      <div className="absolute top-4 right-4 bg-white p-3 rounded-lg shadow-md border border-gray-200">
        <h4 className="font-semibold text-xs mb-2">Heatmap de Consumo</h4>
        <div className="space-y-2">
          <div className="mt-2 pt-2">
            <p className="text-[10px] text-gray-600 mb-1">Intensidade</p>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-yellow-200" />
              <span className="text-[9px] text-gray-500">Baixa</span>
              <div className="w-3 h-3 bg-orange-400" />
              <span className="text-[9px] text-gray-500">Média</span>
              <div className="w-3 h-3 bg-red-600" />
              <span className="text-[9px] text-gray-500">Alta</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function PortugalMap(props: PortugalMapProps) {
  return (
    <MapErrorBoundary>
      <InnerPortugalMap {...props} />
    </MapErrorBoundary>
  );
}
