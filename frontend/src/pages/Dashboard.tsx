import { useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Pill, Building2, Calendar, TrendingUp } from "lucide-react";
import Layout from "@/components/Layout";
import StatsCard from "@/components/StatsCard";
import RegionDistributionChart from "@/components/RegionDistributionChart";
import { PortugalMap, Hospital } from "@/components/PortugalMap";
import { parseCSV, aggregateByHospital } from "@/utils/csvParser";
import { useDistribution } from "@/context/DistributionContext";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { apiService } from "@/services/api";

type HeatmapPeriod = "1m" | "3m" | "1y";

interface DashboardHospital extends Hospital {
  totalConsultations: number;
  monthlyAvgConsultations: number;
  carbapenemsMonthlyAvg?: number;
}

const Dashboard = () => {
  const navigate = useNavigate();
  const { distributions, hospitals, isBackendConnected, apiHospitals } = useDistribution();
  const [hospitalMapData, setHospitalMapData] = useState<DashboardHospital[]>([]);
  const [selectedHospital, setSelectedHospital] = useState<Hospital | null>(null);
  const [heatmapPeriod, setHeatmapPeriod] = useState<HeatmapPeriod>("1m");
  const [modelEfficiency, setModelEfficiency] = useState<number | null>(null);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);

  // Carregar métricas do modelo
  useEffect(() => {
    loadModelMetrics();
  }, []);

  async function loadModelMetrics() {
    try {
      setIsLoadingMetrics(true);
      const metrics = await apiService.getModelMetrics();
      setModelEfficiency(metrics.metrics.efficiency_percentage);
      console.log("📊 Métricas do modelo carregadas:", metrics);
    } catch (error) {
      console.error("Erro ao carregar métricas do modelo:", error);
      // Não definir valor para mostrar "carregando..."
    } finally {
      setIsLoadingMetrics(false);
    }
  }

  // Carregar dados dos hospitais do CSV
  useEffect(() => {
    loadHospitalData();
  }, [apiHospitals]);

  async function loadHospitalData() {
    try {
      const response = await fetch('/01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv');
      const csvText = await response.text();
      const parsed = parseCSV(csvText);
      const aggregated = aggregateByHospital(parsed);

      const periods = new Set(parsed.map((p) => p.period));
      const monthsCount = periods.size || 1;

      const hospitalData: DashboardHospital[] = aggregated.map((h) => {
        const totalConsultations = h.totalConsultations;
        const monthlyAvgConsultations = totalConsultations / monthsCount;

        const apiInfo = apiHospitals?.find((apiH) => apiH.name === h.name);
        const apiConsumption = apiInfo?.avg_monthly_consumption;

        // Usar dados reais se disponíveis E > 0, caso contrário usar estimativa
        const carbapenemsMonthlyAvg = (apiConsumption && apiConsumption > 0)
          ? apiConsumption
          : monthlyAvgConsultations * 0.1;

        return {
          name: h.name,
          lat: h.lat,
          lon: h.lon,
          region: h.region,
          consultations: totalConsultations,
          totalConsultations,
          monthlyAvgConsultations,
          carbapenemsMonthlyAvg,
        };
      });

      setHospitalMapData(hospitalData);

      // Debug: mostrar distribuição de dados reais vs estimados
      const withRealData = hospitalData.filter(h => {
        const apiInfo = apiHospitals?.find(a => a.name === h.name);
        return apiInfo?.avg_monthly_consumption && apiInfo.avg_monthly_consumption > 0;
      });
      console.log(`📊 Carbapenemes: ${withRealData.length}/${hospitalData.length} hospitais com dados reais`);
      console.log('📋 Amostra (consultas vs carbapenemes):');
      hospitalData.slice(0, 5).forEach(h => {
        console.log(`  ${h.name.substring(0, 30)}: ${h.monthlyAvgConsultations.toFixed(1)} consultas → ${h.carbapenemsMonthlyAvg.toFixed(2)} carbapenemes`);
      });
    } catch (error) {
      console.error('Erro ao carregar dados dos hospitais:', error);
    }
  }

  const stats = useMemo(() => {
    const totalHospitals = hospitals.length;
    const estimatedMonthlyConsumption = totalHospitals > 0 ? totalHospitals * 45 : 4050; // Média estimada
    const currentMonth = new Date().toLocaleDateString('pt-PT', { month: 'long', year: 'numeric' });
    console.log("📊 Dashboard stats - Hospital count:", totalHospitals);

    return {
      totalConsumption: estimatedMonthlyConsumption,
      hospitalCount: totalHospitals,
      currentPeriod: currentMonth,
      optimizationScore: modelEfficiency, // Usar métrica real do modelo
      trend: 12.3,
      trendPositive: true,
    };
  }, [hospitals, modelEfficiency]);

  const visualHospitals: Hospital[] = useMemo(() => {
    const hospitals = hospitalMapData.map((h) => {
      const base = h.carbapenemsMonthlyAvg ?? h.monthlyAvgConsultations * 0.1;

      let value: number;
      switch (heatmapPeriod) {
        case "3m":
          value = base * 3;
          break;
        case "1y":
          value = base * 12;
          break;
        case "1m":
        default:
          value = base;
      }

      return {
        name: h.name,
        lat: h.lat,
        lon: h.lon,
        region: h.region,
        consultations: value,
      };
    });

    if (hospitals.length > 0) {
      const sampleHospital = hospitals[0];
      console.log(`🏥 Amostra (${heatmapPeriod}): ${sampleHospital.name.substring(0, 30)} = ${sampleHospital.consultations.toFixed(2)} unidades`);
    }

    return hospitals;
  }, [hospitalMapData, heatmapPeriod]);

  const maxReference = useMemo(() => {
    if (hospitalMapData.length === 0) return 1;

    const maxMonthlyCarbapenems = Math.max(
      1,
      ...hospitalMapData.map((h) => h.carbapenemsMonthlyAvg || 0)
    );

    // Sempre usar o máximo MENSAL como referência
    // Assim quando o período aumenta (3m, 1y), os valores sobem mas o max não,
    // fazendo o mapa ficar progressivamente mais vermelho
    console.log(`📊 Max mensal fixo: ${maxMonthlyCarbapenems.toFixed(2)} (período: ${heatmapPeriod})`);

    return maxMonthlyCarbapenems; // sempre o valor mensal
  }, [hospitalMapData, heatmapPeriod]);
  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Overview of drug distribution across all hospitals
            {isBackendConnected ? " (Real Data)" : " (Mock Data)"}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Consumo Mensal Estimado"
            value={`${stats.totalConsumption.toLocaleString()} unidades`}
            icon={<Pill className="w-6 h-6" />}
            variant="primary"
            trend={{ value: stats.trend, isPositive: stats.trendPositive }}
          />
          <StatsCard
            title="Hospitais Ativos"
            value={stats.hospitalCount}
            icon={<Building2 className="w-6 h-6" />}
          />
          <StatsCard
            title="Período Atual"
            value={stats.currentPeriod}
            icon={<Calendar className="w-6 h-6" />}
          />
          <div onClick={() => navigate('/metrics')} className="cursor-pointer">
            <StatsCard
              title="Eficiência do Algoritmo"
              value={
                isLoadingMetrics
                  ? "Carregando..."
                  : stats.optimizationScore !== null
                    ? `${stats.optimizationScore.toFixed(2)}%`
                    : "N/A"
              }
              icon={<TrendingUp className="w-6 h-6" />}
              variant="success"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <RegionDistributionChart />

          {/* Mapa de Portugal com Hospitais */}
          <div className="bg-card rounded-lg border shadow-sm p-6">
            <div className="flex items-center justify-between mb-4 gap-4">
              <h2 className="text-xl font-semibold">Consumo de Carbapenemes</h2>
              <div className="flex flex-col gap-1">
                <Label className="text-[11px]">Período</Label>
                <Select
                  value={heatmapPeriod}
                  onValueChange={(v) => setHeatmapPeriod(v as HeatmapPeriod)}
                >
                  <SelectTrigger className="h-8 w-32 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="z-[9999]">
                    <SelectItem value="1m">Último mês</SelectItem>
                    <SelectItem value="3m">Últimos 3 meses</SelectItem>
                    <SelectItem value="1y">Último ano</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {visualHospitals.length > 0 ? (
              <div className="relative" style={{ height: '500px' }}>
                <PortugalMap
                  hospitals={visualHospitals}
                  onHospitalClick={setSelectedHospital}
                  metric="carbapenems"
                  maxReference={maxReference}
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                  <p className="mt-2 text-sm text-muted-foreground">A carregar mapa...</p>
                </div>
              </div>
            )}

            {selectedHospital && (
              <div className="mt-4 p-4 bg-muted rounded-lg">
                <h3 className="font-semibold text-sm mb-2">Hospital Selecionado</h3>
                <p className="text-xs text-muted-foreground truncate">{selectedHospital.name}</p>
                <p className="text-xs mt-1">
                  <strong>Região:</strong> {selectedHospital.region}
                </p>
                {selectedHospital.consultations && (
                  <p className="text-xs mt-1">
                    <strong>Carbapenemes ({heatmapPeriod === "1m" ? "último mês" : heatmapPeriod === "3m" ? "últimos 3 meses" : "último ano"}):</strong> {selectedHospital.consultations.toLocaleString()} unidades
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Dashboard;
