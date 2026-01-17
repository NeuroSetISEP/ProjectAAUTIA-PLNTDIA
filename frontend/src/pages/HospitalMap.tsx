import { useState, useEffect } from 'react';
import { PortugalMap, Hospital } from '../components/PortugalMap';
import { parseCSV, aggregateByHospital } from '../utils/csvParser';
import Layout from '@/components/Layout';
import { useDistribution } from '@/context/DistributionContext';

export default function HospitalMap() {
  const { apiHospitals } = useDistribution();
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedHospital, setSelectedHospital] = useState<Hospital | null>(null);

  useEffect(() => {
    loadHospitalData();
  }, [apiHospitals]);

  async function loadHospitalData() {
    try {
      const response = await fetch('/01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv');
      const csvText = await response.text();

      const parsed = parseCSV(csvText);
      const aggregated = aggregateByHospital(parsed);

      const hospitalData: Hospital[] = aggregated.map(h => {
        // Buscar dados de carbapenemes da API se disponível
        const apiInfo = apiHospitals?.find((apiH) => apiH.name === h.name);
        const carbapenemConsumption = (apiInfo?.avg_monthly_consumption && apiInfo.avg_monthly_consumption > 0)
          ? apiInfo.avg_monthly_consumption
          : 0;

        return {
          name: h.name,
          lat: h.lat,
          lon: h.lon,
          region: h.region,
          consultations: carbapenemConsumption, // usar carbapenemes em vez de consultas
        };
      });

      setHospitals(hospitalData);
    } catch (error) {
      console.error('Erro ao carregar dados dos hospitais:', error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Layout>
      <div className="p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Mapa de Hospitais</h1>
          <p className="text-muted-foreground mt-1">
            Visualização geográfica dos hospitais portugueses com consumo de carbapenemes
          </p>
        </header>

        {loading ? (
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
              <p className="mt-4 text-muted-foreground">A carregar dados...</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Mapa */}
            <div className="lg:col-span-2 bg-card rounded-lg border shadow-sm p-6">
              <div style={{ height: '600px' }}>
                <PortugalMap
                  hospitals={hospitals}
                  onHospitalClick={setSelectedHospital}
                  metric="carbapenems"
                />
              </div>
            </div>

            {/* Painel lateral com estatísticas */}
            <div className="space-y-4">
              {/* Card de estatísticas gerais */}
              <div className="bg-card rounded-lg border shadow-sm p-4">
                <h2 className="text-lg font-semibold mb-4">Estatísticas</h2>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-muted-foreground">Total de Hospitais</p>
                    <p className="text-2xl font-bold text-primary">{hospitals.length}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Consumo Total Mensal</p>
                    <p className="text-2xl font-bold text-accent">
                      {hospitals.reduce((sum, h) => sum + (h.consultations || 0), 0).toLocaleString()} un.
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Média por Hospital</p>
                    <p className="text-2xl font-bold text-secondary">
                      {Math.round(hospitals.reduce((sum, h) => sum + (h.consultations || 0), 0) / hospitals.length).toLocaleString()} un.
                    </p>
                  </div>
                </div>
              </div>

              {/* Card do hospital selecionado */}
              {selectedHospital && (
                <div className="bg-muted border rounded-lg shadow-sm p-4">
                  <h2 className="text-lg font-semibold mb-3">Hospital Selecionado</h2>
                  <div className="space-y-2">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Nome</p>
                      <p className="text-sm font-medium">{selectedHospital.name}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Região</p>
                      <p className="text-sm">{selectedHospital.region}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Consumo Mensal de Carbapenemes</p>
                      <p className="text-sm font-semibold">{(selectedHospital.consultations || 0).toLocaleString()} unidades</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Coordenadas</p>
                      <p className="text-xs font-mono">
                        {selectedHospital.lat.toFixed(4)}, {selectedHospital.lon.toFixed(4)}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Top 5 hospitais */}
              <div className="bg-card rounded-lg border shadow-sm p-4">
                <h2 className="text-lg font-semibold mb-3">Top 5 Hospitais</h2>
                <div className="space-y-2">
                  {[...hospitals]
                    .sort((a, b) => (b.consultations || 0) - (a.consultations || 0))
                    .slice(0, 5)
                    .map((h, idx) => (
                      <div
                        key={idx}
                        className="p-2 bg-muted rounded border cursor-pointer hover:bg-accent/10 transition-colors"
                        onClick={() => setSelectedHospital(h)}
                      >
                        <p className="text-xs font-medium truncate">{h.name.split(',')[0]}</p>
                        <p className="text-xs text-muted-foreground">{(h.consultations || 0).toLocaleString()} unidades/mês</p>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
